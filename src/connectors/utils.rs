use bytes::Bytes;
use reqwest::IntoUrl;

pub async fn download_resource<T: IntoUrl>(url: T) -> anyhow::Result<Bytes> {
    Ok(reqwest::get(url).await?.bytes().await?)
}
