use std::{collections::HashMap, path::PathBuf, sync::{Arc, RwLock}};

use anndata::{backend::{AttributeOp, DatasetOp, DynArrayView, GroupOp, ScalarType, StoreOp}, data::{DynArray, Shape}, Backend, Data};
use anyhow::Ok;
use ndarray::{Array, ArrayView};


// In memory backend for anndata representation, temporary!
pub struct InMemory {
    root: Arc<RwLock<InMemoryNode>>
}

pub struct File {
    root: Arc<RwLock<InMemoryNode>>,
    filename: PathBuf
}

pub struct Group {
    node: Arc<RwLock<InMemoryNode>>,
    path: PathBuf
}

pub struct Dataset {
    data: Arc<RwLock<DynArray>>,
    dtype: ScalarType,
    shape: Shape
}

enum InMemoryNode {
    Group(HashMap<String, InMemoryNode>),
    Dataset(Dataset)
}

impl Backend for InMemory {
    const NAME: &'static str = "memory";

    type Store = File;

    type Group = Group;

    type Dataset = Dataset;

    fn create<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self::Store> {
        let root = Arc::new(RwLock::new(InMemoryNode::Group(HashMap::new())));
        Ok(File {
            root: root.clone(),
            filename: path.as_ref().to_path_buf(),
        })
    }

    fn open<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self::Store> {
        Self::create(path)
    }

    fn open_rw<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self::Store> {
        Self::create(path)
    }
}

impl StoreOp<InMemory> for File {
    fn filename(&self) -> PathBuf {
        self.filename.clone()
    }

    fn close(self) -> anyhow::Result<()> {
        Ok(())
    }
}

impl GroupOp<InMemory> for File {
    fn list(&self) -> anyhow::Result<Vec<String>> {
        if let InMemoryNode::Group(map) = &*self.root.read().unwrap() {
            Ok(map.keys().cloned().collect())
        } else {
            anyhow::bail!("Root is not a group")
        }
    }

    fn create_group(&self, name: &str) -> anyhow::Result<<InMemory as Backend>::Group> {
        let new_group = InMemoryNode::Group(HashMap::new());
        if let InMemoryNode::Group(ref mut map) = &mut *self.root.write().unwrap() {
            map.insert(name.to_string(), new_group);
            Ok(
                Group {
                    node: Arc::new(RwLock::new(InMemoryNode::Group(HashMap::new()))),
                    path: PathBuf::from(name)
                }
            )
        } else {
            anyhow::bail!("Root is not a group!")
        }
    }

    fn open_group(&self, name: &str) -> anyhow::Result<<InMemory as Backend>::Group> {
        if let InMemoryNode::Group(map) = &*self.root.read().unwrap() {
            if let Some(InMemoryNode::Group(_)) = map.get(name) {
                Ok(Group {
                    node: Arc::new(RwLock::new(map[name].clone())),
                    path: PathBuf::from(name),
                })
            } else {
                anyhow::bail!("Group not found")
            }
        } else {
            anyhow::bail!("Root is not a group")
        }
    }

    fn new_dataset<T: anndata::backend::BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: anndata::backend::WriteConfig,
    ) -> anyhow::Result<<InMemory as Backend>::Dataset> {
        let empty_array = match T::DTYPE {
            ScalarType::I8 => DynArray::I8(Array::default(shape.as_ref())),
            ScalarType::I16 => DynArray::I16(Array::default(shape.as_ref())),
            ScalarType::I32 => DynArray::I32(Array::default(shape.as_ref())),
            ScalarType::I64 => DynArray::I64(Array::default(shape.as_ref())),
            ScalarType::U8 => DynArray::U8(Array::default(shape.as_ref())),
            ScalarType::U16 => DynArray::U16(Array::default(shape.as_ref())),
            ScalarType::U32 => DynArray::U32(Array::default(shape.as_ref())),
            ScalarType::U64 => DynArray::U64(Array::default(shape.as_ref())),
            ScalarType::Usize => DynArray::Usize(Array::default(shape.as_ref())),
            ScalarType::F32 => DynArray::F32(Array::default(shape.as_ref())),
            ScalarType::F64 => DynArray::F64(Array::default(shape.as_ref())),
            ScalarType::Bool => DynArray::Bool(Array::default(shape.as_ref())),
            ScalarType::String => DynArray::String(Array::default(shape.as_ref())),
        };
        let dataset = Dataset {
            
            data: Arc::new(RwLock::new(empty_array)),
            dtype: T::DTYPE,
            shape: shape.clone(),
        };

        if let InMemoryNode::Group(ref mut map) = &mut *self.root.write().unwrap() {
            map.insert(name.to_string(), InMemoryNode::Dataset(dataset.clone()));
            Ok(dataset)
        } else {
            anyhow::bail!("Root is not a group!")
        }
    }

    fn open_dataset(&self, name: &str) -> anyhow::Result<<InMemory as Backend>::Dataset> {
        if let InMemoryNode::Group(map) = &*self.root.read().unwrap() {
            if let Some(InMemoryNode::Dataset(dataset)) = map.get(name) {
                Ok(dataset.clone())
            } else {
                anyhow::bail!("Dataset has not been found!")
            }
        } else {
            anyhow::bail!("Root is not a group!")
        }
    }

    fn delete(&self, name: &str) -> anyhow::Result<()> {
        if let InMemoryNode::Group(ref mut map) = &mut *self.root.write().unwrap() {
            map.remove(name);
            Ok(())
        } else {
            anyhow::bail!("Root is not a group!")
        }
    }

    fn exists(&self, name: &str) -> anyhow::Result<bool> {
        if let InMemoryNode::Group(map) = &*self.root.read().unwrap() {
            Ok(map.contains_key(name))
        } else {
            anyhow::bail!("Root is not a group!")
        }
    }
}

impl AttributeOp<InMemory> for Group {
    fn store(&self) -> anyhow::Result<File> {
         Ok(
            File {
                root: self.node.clone(),
                filename: self.path.clone(),
            }
         )
    }

    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> anyhow::Result<()>
    where
        A: Into<ndarray::ArrayView<'a, D, Dim>>,
        D: anndata::backend::BackendData,
        Dim: ndarray::Dimension {
        
            let array_view = value.into();
            let shape = array_view.shape().to_vec();
            let dyn_array_view = anndata::backend::BackendData::into_dyn_arr(array_view.into_dyn());
            
            let dyn_array = match dyn_array_view {
                DynArrayView::U8(x) => DynArray::U8(x.to_owned()),
                DynArrayView::U16(x) => DynArray::U16(x.to_owned()),
                DynArrayView::U32(x) => DynArray::U32(x.to_owned()),
                DynArrayView::U64(x) => DynArray::U64(x.to_owned()),
                DynArrayView::Usize(x) => DynArray::Usize(x.to_owned()),
                DynArrayView::I8(x) => DynArray::I8(x.to_owned()),
                DynArrayView::I16(x) => DynArray::I16(x.to_owned()),
                DynArrayView::I32(x) => DynArray::I32(x.to_owned()),
                DynArrayView::I64(x) => DynArray::I64(x.to_owned()),
                DynArrayView::F32(x) => DynArray::F32(x.to_owned()),
                DynArrayView::F64(x) => DynArray::F64(x.to_owned()),
                DynArrayView::Bool(x) => DynArray::Bool(x.to_owned()),
                DynArrayView::String(x) => DynArray::String(x.to_owned()),
            };
    
            let dataset = Dataset {
                data: Arc::new(RwLock::new(dyn_array)),
                dtype: D::DTYPE,
                shape: shape.into(),
            };
            
            if let InMemoryNode::Group(ref mut map) = &mut *self.node.write().unwrap() {
                map.insert(name.to_string(), InMemoryNode::Dataset(dataset));
                Ok(())
            } else {
                anyhow::bail!("Not a group")
            }

    }

    fn read_array_attr<T: anndata::backend::BackendData, D: ndarray::Dimension>(&self, name: &str) -> anyhow::Result<Array<T, D>> {
        if let InMemoryNode::Group(map) = &*self.node.read().unwrap() {
            if let Some(InMemoryNode::Dataset(dataset)) = map.get(name) {
                let array = dataset.data.read().unwrap().clone();
                Ok(anndata::backend::BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
            } else {
                anyhow::bail!("Attribute not found!")
            }
        } else {
            anyhow::bail!("Not a group")
        }
    }
}

impl DatasetOp<InMemory> for Dataset {
    fn dtype(&self) -> anyhow::Result<ScalarType> {
        Ok(self.dtype)
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn reshape(&self, shape: &Shape) -> anyhow::Result<()> {
        let mut data = self.data.write().unwrap();
        let new_shape = (shape[0], shape[1]);
        *data = match *data {
            DynArray::U8(ref arr) => DynArray::U8(arr.clone().into_shape((shape[0], shape[1]))?),
            DynArray::U16(ref arr) => DynArray::U16(arr.clone().into_shape(shape.clone())?),
            DynArray::U32(ref arr) => DynArray::U32(arr.clone().into_shape(shape.clone())?),
            DynArray::U64(ref arr) => DynArray::U64(arr.clone().into_shape(shape.clone())?),
            DynArray::Usize(ref arr) => DynArray::Usize(arr.clone().into_shape(shape.clone())?),
            DynArray::I8(ref arr) => DynArray::I8(arr.clone().into_shape(shape.clone())?),
            DynArray::I16(ref arr) => DynArray::I16(arr.clone().into_shape(shape.clone())?),
            DynArray::I32(ref arr) => DynArray::I32(arr.clone().into_shape(shape.clone())?),
            DynArray::I64(ref arr) => DynArray::I64(arr.clone().into_shape(shape.clone())?),
            DynArray::F32(ref arr) => DynArray::F32(arr.clone().into_shape(shape.clone())?),
            DynArray::F64(ref arr) => DynArray::F64(arr.clone().into_shape(shape.clone())?),
            DynArray::Bool(ref arr) => DynArray::Bool(arr.clone().into_shape(shape.clone())?),
            DynArray::String(ref arr) => DynArray::String(arr.clone().into_shape(shape.clone())?),
        };
        self.shape = shape.clone();
        Ok(())
    }

    fn read_array_slice<T: anndata::backend::BackendData, S, D>(&self, selection: &[S]) -> anyhow::Result<Array<T, D>>
    where
        S: AsRef<anndata::data::SelectInfoElem>,
        D: ndarray::Dimension {
        todo!()
    }

    fn write_array_slice<'a, A, S, T, D>(
        &self,
        data: A,
        selection: &[S],
    ) -> anyhow::Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        T: anndata::backend::BackendData,
        S: AsRef<anndata::data::SelectInfoElem>,
        D: ndarray::Dimension {
        todo!()
    }
}

