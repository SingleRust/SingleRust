use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    vec::IntoIter,
};

use anndata::{
    backend::{DataType, ScalarType},
    container::{Axis, Dim},
    data::{index::Index, ArrayChunk, DataFrameIndex, SelectInfoElem},
    AnnDataOp, ArrayData, ArrayElemOp, ArrayOp, AxisArraysOp, Data, ElemCollectionOp, HasShape,
    ReadArrayData, ReadData, WriteArrayData, WriteData,
};
use anyhow::anyhow;
use polars::prelude::{DataFrame, NamedFrom};

/// In-memory representation of AnnData
pub struct InMemoryAnnData {
    pub(crate) n_obs: Dim,
    pub(crate) n_vars: Dim,
    x: Arc<RwLock<InnerElemInMemory>>,
    obs: Arc<RwLock<DataFrame>>,
    obsm: AnnotationMatrix,
    obsp: AnnotationMatrix,
    var: Arc<RwLock<DataFrame>>,
    varm: AnnotationMatrix,
    varp: AnnotationMatrix,
    uns: InMemoryElemCollection,
    layers: AnnotationMatrix,
}

/// Internal representation of array elements in memory
pub struct InnerElemInMemory {
    dtype: DataType,
    container: ArrayData,
}

impl ArrayElemOp for InnerElemInMemory {
    type ArrayIter<T> = IntoIter<(T, usize, usize)>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn shape(&self) -> Option<anndata::data::Shape> {
        Some(self.container.shape())
    }

    fn get<D>(&self) -> anyhow::Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        D::try_from(self.container.clone())
            .map(Some)
            .map_err(Into::into)
    }

    fn slice<D, S>(&self, slice: S) -> anyhow::Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let selected = self.container.select(slice.as_ref());
        D::try_from(selected).map(Some).map_err(Into::into)
    }

    fn iter<T>(&self, chunk_size: usize) -> Self::ArrayIter<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let total_size = self.container.shape()[0];
        (0..total_size)
            .step_by(chunk_size)
            .filter_map(|start| {
                let end = std::cmp::min(start + chunk_size, total_size);
                let chunk = self.container.select(&[SelectInfoElem::from(start..end)]);
                T::try_from(chunk).ok().map(|t| (t, start, end))
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl Clone for InnerElemInMemory {
    fn clone(&self) -> Self {
        Self {
            dtype: self.dtype,
            container: self.container.clone(),
        }
    }
}

/// Represents a matrix of annotations
pub struct AnnotationMatrix {
    data: Arc<RwLock<HashMap<String, InnerElemInMemory>>>,
    axis: Axis,
    dim1: usize,
    dim2: Option<usize>,
}

impl AxisArraysOp for &AnnotationMatrix {
    type ArrayElem = InnerElemInMemory;

    fn keys(&self) -> Vec<String> {
        self.data.read().unwrap().keys().cloned().collect()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.data.read().unwrap().get(key).cloned()
    }

    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> anyhow::Result<()> {
        let dtype = data.data_type();
        let shape = data.shape();

        // Perform dimensionality checks based on the axis type
        match self.axis {
            Axis::Row => {
                if shape[0] != self.dim1 {
                    return Err(anyhow!("Data shape {:?} does not match expected row dimension {}", shape, self.dim1));
                }
            },
            Axis::RowColumn => {
                if shape[0] != self.dim1 || shape[1] != self.dim2.unwrap() {
                    return Err(anyhow!("Data shape {:?} does not match expected dimensions ({}, {})", 
                                       shape, self.dim1, self.dim2.unwrap()));
                }
            },
            Axis::Pairwise => {
                if shape[0] != self.dim1 || shape[1] != self.dim1 {
                    return Err(anyhow!("Data shape {:?} does not match expected pairwise dimensions ({}, {})", 
                                       shape, self.dim1, self.dim1));
                }
            },
        }

        let container = data.into();
        let elem = InnerElemInMemory { dtype, container };
        self.data.write().unwrap().insert(key.to_string(), elem);
        Ok(())
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> anyhow::Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>
    {
        let merged = ArrayData::vstack(data.map(Into::into))?;
        
        // Perform dimensionality check on the merged data
        let shape = merged.shape();
        match self.axis {
            Axis::Row => {
                if shape[0] != self.dim1 {
                    return Err(anyhow!("Merged data shape {:?} does not match expected row dimension {}", shape, self.dim1));
                }
            },
            Axis::RowColumn => {
                if shape[1] != self.dim2.unwrap() {
                    return Err(anyhow!("Merged data shape {:?} does not match expected column dimension {}", 
                                       shape, self.dim2.unwrap()));
                }
            },
            Axis::Pairwise => {
                if shape[1] != self.dim1 {
                    return Err(anyhow!("Merged data shape {:?} does not match expected pairwise dimension {}", 
                                       shape, self.dim1));
                }
            },
        }

        self.add(key, merged)
    }

    fn remove(&self, key: &str) -> anyhow::Result<()> {
        self.data.write().unwrap().remove(key);
        Ok(())
    }
}

/// Collection of elements stored in memory
pub struct InMemoryElemCollection {
    data: Arc<RwLock<HashMap<String, InnerElemInMemory>>>,
}

impl ElemCollectionOp for &InMemoryElemCollection {
    fn keys(&self) -> Vec<String> {
        self.data.read().unwrap().keys().cloned().collect()
    }

    fn get_item<D>(&self, key: &str) -> anyhow::Result<Option<D>>
    where
        D: ReadData + Into<anndata::Data> + TryFrom<anndata::Data> + Clone,
        <D as TryFrom<anndata::Data>>::Error: Into<anyhow::Error>
    {
        self.data.read().unwrap()
            .get(key)
            .map(|data| D::try_from(Data::ArrayData(data.container.clone())))
            .transpose()
            .map_err(Into::into)
    }

    fn add<D: anndata::WriteData + Into<anndata::Data>>(&self, key: &str, data: D) -> anyhow::Result<()> {
        if let Data::ArrayData(array_data) = data.into() {
            let elem = InnerElemInMemory {
                dtype: array_data.data_type(),
                container: array_data,
            };
            self.data.write().unwrap().insert(key.to_string(), elem);
            Ok(())
        } else {
            Err(anyhow!("Only ArrayData is supported in InMemoryElemCollection"))
        }
    }

    fn remove(&self, key: &str) -> anyhow::Result<()> {
        self.data.write().unwrap().remove(key);
        Ok(())
    }
}

impl AnnDataOp for InMemoryAnnData {
    type X = InnerElemInMemory;
    type AxisArraysRef<'a> = &'a AnnotationMatrix;
    type ElemCollectionRef<'a> = &'a InMemoryElemCollection;

    fn x(&self) -> Self::X {
        self.x.read().unwrap().clone()
    }

    fn set_x_from_iter<I, D>(&self, iter: I) -> anyhow::Result<()>
    where
        I: Iterator<Item = D>,
        D: anndata::data::ArrayChunk + Into<anndata::ArrayData>,
    {
        let merged = ArrayData::vstack(iter.map(Into::into))?;
        self.set_x(merged)
    }

    fn set_x<D: anndata::WriteArrayData + Into<anndata::ArrayData> + anndata::HasShape>(
        &self,
        data: D,
    ) -> anyhow::Result<()> {
        let shape = data.shape();
        self.n_obs.try_set(shape[0])?;
        self.n_vars.try_set(shape[1])?;
        
        let mut x_val = self.x.write().unwrap();
        x_val.dtype = data.data_type();
        x_val.container = data.into();
        Ok(())
    }

    fn del_x(&self) -> anyhow::Result<()> {
        let mut x_val = self.x.write().unwrap();
        *x_val = InnerElemInMemory {
            dtype: DataType::Array(ScalarType::F64),
            container: ArrayData::Array(anndata::data::DynArray::F64(ndarray::Array2::zeros((0, 0)).into_dyn())),
        };
        Ok(())
    }

    fn n_obs(&self) -> usize {
        self.n_obs.get()
    }

    fn n_vars(&self) -> usize {
        self.n_vars.get()
    }

    fn set_n_obs(&self, n: usize) -> anyhow::Result<()> {
        self.n_obs.try_set(n)
    }

    fn set_n_vars(&self, n: usize) -> anyhow::Result<()> {
        self.n_vars.try_set(n)
    }

    fn obs_names(&self) -> DataFrameIndex {
        let obs = self.obs.read().unwrap();
        let index_col = obs.column("index").unwrap();
        let index_vec: Vec<String> = index_col.str().unwrap()
            .into_iter()
            .filter_map(|opt_s| opt_s.map(str::to_string))
            .collect();
        DataFrameIndex::from(Index::from(index_vec))
    }

    fn var_names(&self) -> DataFrameIndex {
        let var = self.var.read().unwrap();
        let index_col = var.column("index").unwrap();
        let index_vec: Vec<String> = index_col.str().unwrap()
            .into_iter()
            .filter_map(|opt_s| opt_s.map(str::to_string))
            .collect();
        DataFrameIndex::from(Index::from(index_vec))
    }
    
    fn set_obs_names(&self, index: DataFrameIndex) -> anyhow::Result<()> {
        let mut obs = self.obs.write().unwrap();
        let index_series = polars::prelude::Series::new("index", index.into_vec());
        obs.with_column(index_series)?;
        Ok(())
    }
    
    fn set_var_names(&self, index: DataFrameIndex) -> anyhow::Result<()> {
        let mut var = self.var.write().unwrap();
        let index_series = polars::prelude::Series::new("index", index.into_vec());
        var.with_column(index_series)?;
        Ok(())
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> anyhow::Result<Vec<usize>> {
        let obs = self.obs.read().unwrap();
        let index = obs.column("index").unwrap();
        let index_vec: Vec<String> = index.str().unwrap()
            .into_iter()
            .filter_map(|opt_s| opt_s.map(str::to_string))
            .collect();
        names.into_iter()
            .map(|name| index_vec.iter().position(|s| s == name)
                .ok_or_else(|| anyhow!("Name '{}' not found in obs_names", name)))
            .collect()
    }

    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> anyhow::Result<Vec<usize>> {
        let var = self.var.read().unwrap();
        let index = var.column("index").unwrap();
        let index_vec: Vec<String> = index.str().unwrap()
            .into_iter()
            .filter_map(|opt_s| opt_s.map(str::to_string))
            .collect();
        names.into_iter()
            .map(|name| index_vec.iter().position(|s| s == name)
                .ok_or_else(|| anyhow!("Name '{}' not found in var_names", name)))
            .collect()
    }

    fn read_obs(&self) -> anyhow::Result<DataFrame> {
        Ok(self.obs.read().unwrap().clone())
    }

    fn read_var(&self) -> anyhow::Result<DataFrame> {
        Ok(self.var.read().unwrap().clone())
    }

    fn set_obs(&self, obs: polars::frame::DataFrame) -> anyhow::Result<()> {
        *self.obs.write().unwrap() = obs;
        Ok(())
    }

    fn set_var(&self, var: DataFrame) -> anyhow::Result<()> {
        *self.var.write().unwrap() = var;
        Ok(())
    }

    fn del_obs(&self) -> anyhow::Result<()> {
        *self.obs.write().unwrap() = DataFrame::default();
        Ok(())
    }

    fn del_var(&self) -> anyhow::Result<()> {
        *self.var.write().unwrap() = DataFrame::default();
        Ok(())
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        &self.uns
    }

    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        &self.obsm
    }

    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        &self.obsp
    }

    fn varm(&self) -> Self::AxisArraysRef<'_> {
        &self.varm
    }

    fn varp(&self) -> Self::AxisArraysRef<'_> {
        &self.varp
    }

    fn layers(&self) -> Self::AxisArraysRef<'_> {
        &self.layers
    }

    fn del_uns(&self) -> anyhow::Result<()> {
        self.uns.data.write().unwrap().clear();
        Ok(())
    }

    fn del_obsm(&self) -> anyhow::Result<()> {
        self.obsm.data.write().unwrap().clear();
        Ok(())
    }

    fn del_obsp(&self) -> anyhow::Result<()> {
        self.obsp.data.write().unwrap().clear();
        Ok(())
    }

    fn del_varm(&self) -> anyhow::Result<()> {
        self.varm.data.write().unwrap().clear();
        Ok(())
    }

    fn del_varp(&self) -> anyhow::Result<()> {
        self.varp.data.write().unwrap().clear();
        Ok(())
    }

    fn del_layers(&self) -> anyhow::Result<()> {
        self.layers.data.write().unwrap().clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use polars::series::Series;

    fn create_test_anndata() -> InMemoryAnnData {
        let n_obs = Dim::new(100);
        let n_vars = Dim::new(50);
        let x = Arc::new(RwLock::new(InnerElemInMemory {
            dtype: anndata::backend::DataType::Array(ScalarType::F64),
            container: ArrayData::Array(anndata::data::DynArray::F64(Array2::zeros((100, 50)).into_dyn())),
        }));
        
        let obs = Arc::new(RwLock::new(DataFrame::new(vec![
            Series::new("index", (0..100).map(|i| format!("obs_{}", i)).collect::<Vec<String>>()),
        ]).unwrap()));
        
        let columns = vec![
            Series::new("index", (0..50).map(|i| format!("var_{}", i)).collect::<Vec<String>>()),
        ];
        let var = Arc::new(RwLock::new(DataFrame::new(columns).unwrap()));

        InMemoryAnnData {
            n_obs,
            n_vars,
            x,
            obs,
            var,
            obsm: AnnotationMatrix {
                data: Arc::new(RwLock::new(HashMap::new())),
                axis: Axis::Row,
                dim1: 100,
                dim2: None,
            },
            obsp: AnnotationMatrix {
                data: Arc::new(RwLock::new(HashMap::new())),
                axis: Axis::Pairwise,
                dim1: 100,
                dim2: None,
            },
            varm: AnnotationMatrix {
                data: Arc::new(RwLock::new(HashMap::new())),
                axis: Axis::Row,
                dim1: 50,
                dim2: None,
            },
            varp: AnnotationMatrix {
                data: Arc::new(RwLock::new(HashMap::new())),
                axis: Axis::Pairwise,
                dim1: 50,
                dim2: None,
            },
            uns: InMemoryElemCollection {
                data: Arc::new(RwLock::new(HashMap::new())),
            },
            layers: AnnotationMatrix {
                data: Arc::new(RwLock::new(HashMap::new())),
                axis: Axis::RowColumn,
                dim1: 100,
                dim2: Some(50),
            },
        }
    }

    #[test]
    fn test_dimensions() {
        let adata = create_test_anndata();
        assert_eq!(adata.n_obs(), 100);
        assert_eq!(adata.n_vars(), 50);
    }

    #[test]
    fn test_obs_var_operations() {
        let adata = create_test_anndata();

        // Test read_obs and read_var
        let obs = adata.read_obs().unwrap();
        let var = adata.read_var().unwrap();
        assert_eq!(obs.shape(), (100, 1));
        assert_eq!(var.shape(), (50, 1));

        // Test set_obs and set_var
        let new_obs = DataFrame::new(vec![
            Series::new("index", (0..100).map(|i| format!("new_obs_{}", i)).collect::<Vec<String>>()),
            Series::new("value", (0..100).collect::<Vec<i32>>()),
        ]).unwrap();
        adata.set_obs(new_obs.clone()).unwrap();
        assert_eq!(adata.read_obs().unwrap(), new_obs);

        let new_var = DataFrame::new(vec![
            Series::new("index", (0..50).map(|i| format!("new_var_{}", i)).collect::<Vec<String>>()),
            Series::new("value", (0..50).collect::<Vec<i32>>()),
        ]).unwrap();
        adata.set_var(new_var.clone()).unwrap();
        assert_eq!(adata.read_var().unwrap(), new_var);

        // Test obs_ix and var_ix
        let obs_indices = adata.obs_ix(["new_obs_0", "new_obs_50", "new_obs_99"]).unwrap();
        assert_eq!(obs_indices, vec![0, 50, 99]);

        let var_indices = adata.var_ix(["new_var_0", "new_var_25", "new_var_49"]).unwrap();
        assert_eq!(var_indices, vec![0, 25, 49]);

        // Test del_obs and del_var
        adata.del_obs().unwrap();
        adata.del_var().unwrap();
        assert_eq!(adata.read_obs().unwrap().shape(), (0, 0));
        assert_eq!(adata.read_var().unwrap().shape(), (0, 0));
    }

    #[test]
    fn test_annotation_matrices() {
        let adata = create_test_anndata();

        // Test obsm
        let obsm = adata.obsm();
        obsm.add("test_obsm", Array2::<f64>::ones((100, 10))).unwrap();
        assert_eq!(obsm.keys(), vec!["test_obsm"]);
        let retrieved_obsm: Array2<f64> = obsm.get("test_obsm").unwrap().get().unwrap().unwrap();
        assert_eq!(retrieved_obsm.shape(), &[100, 10]);

        // Test varm
        let varm = adata.varm();
        varm.add("test_varm", Array2::<f64>::ones((50, 5))).unwrap();
        assert_eq!(varm.keys(), vec!["test_varm"]);
        let retrieved_varm: Array2<f64> = varm.get("test_varm").unwrap().get().unwrap().unwrap();
        assert_eq!(retrieved_varm.shape(), &[50, 5]);

        // Test layers
        let layers = adata.layers();
        layers.add("test_layer", Array2::<f64>::ones((100, 50))).unwrap();
        assert_eq!(layers.keys(), vec!["test_layer"]);
        let retrieved_layer: Array2<f64> = layers.get("test_layer").unwrap().get().unwrap().unwrap();
        assert_eq!(retrieved_layer.shape(), &[100, 50]);
    }

    #[test]
    fn test_uns() {
        let adata = create_test_anndata();
        let uns = adata.uns();

        uns.add("test_uns", Array2::<f64>::ones((5, 5))).unwrap();
        assert_eq!(uns.keys(), vec!["test_uns"]);
        let retrieved_uns: Array2<f64> = uns.get_item("test_uns").unwrap().unwrap();
        assert_eq!(retrieved_uns.shape(), &[5, 5]);

        uns.remove("test_uns").unwrap();
        assert!(uns.keys().is_empty());
    }
}