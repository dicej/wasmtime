#![deny(warnings)]

wasmtime::component::bindgen!({
    trappable_imports: true,
    path: "../wasi-http/src/p3/wit",
    interfaces: "
      import wasi:http/types@0.3.0-draft;
      import wasi:http/handler@0.3.0-draft;
    ",
    concurrent_imports: true,
    async: {
        only_imports: [
            "wasi:http/types@0.3.0-draft#[static]request.new",
            "wasi:http/types@0.3.0-draft#[static]response.new",
            "wasi:http/handler@0.3.0-draft#[async]handle",
        ]
    },
    with: {
        "wasi:http/types/request": Request,
        "wasi:http/types/request-options": RequestOptions,
        "wasi:http/types/response": Response,
        "wasi:http/types/fields": Fields,
    },
});

use {
    anyhow::anyhow,
    bytes::BytesMut,
    std::{fmt, future::Future, marker, mem},
    wasi::http::types::{ErrorCode, HeaderError, Method, RequestOptionsError, Scheme},
    wasmtime::component::{
        Accessor, AccessorTask, FutureReader, HasData, HostFuture, HostStream, Linker, Resource,
        ResourceTable, StreamReader,
    },
};

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Scheme::Http => "http",
                Scheme::Https => "https",
                Scheme::Other(s) => s,
            }
        )
    }
}

pub trait WasiHttpViewConcurrent: Send + 'static {
    type View<'a>: WasiHttpView;

    fn send_request<T: 'static>(
        accessor: &Accessor<T, WasiHttp<Self>>,
        request: Resource<Request>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<Response>, ErrorCode>>> + Send + Sync;
}

pub trait WasiHttpView: Send {
    fn table(&mut self) -> &mut ResourceTable;
}

impl<T: WasiHttpView + ?Sized> WasiHttpView for &mut T {
    fn table(&mut self) -> &mut ResourceTable {
        (*self).table()
    }
}

struct SendRequestTask<C> {
    request: Resource<Request>,
    _marker: marker::PhantomData<fn() -> C>,
}

impl<T: 'static, C>
    AccessorTask<T, WasiHttp<C>, wasmtime::Result<Result<Resource<Response>, ErrorCode>>>
    for SendRequestTask<C>
where
    C: WasiHttpViewConcurrent,
{
    async fn run(
        self,
        accessor: &Accessor<T, WasiHttp<C>>,
    ) -> wasmtime::Result<Result<Resource<Response>, ErrorCode>> {
        C::send_request(accessor, self.request).await
    }
}

pub struct WasiHttp<C: ?Sized>(marker::PhantomData<C>);

impl<C: ?Sized> HasData for WasiHttp<C>
where
    C: WasiHttpViewConcurrent,
{
    type Data<'a> = WasiHttpImpl<C::View<'a>>;
}

#[repr(transparent)]
pub struct WasiHttpImpl<T>(pub T);

impl<T: WasiHttpView> WasiHttpView for WasiHttpImpl<T> {
    fn table(&mut self) -> &mut ResourceTable {
        self.0.table()
    }
}

#[derive(Clone)]
pub struct Fields(pub Vec<(String, Vec<u8>)>);

#[derive(Default, Copy, Clone)]
pub struct RequestOptions {
    pub connect_timeout: Option<u64>,
    pub first_byte_timeout: Option<u64>,
    pub between_bytes_timeout: Option<u64>,
}

pub struct Request {
    pub method: Method,
    pub scheme: Option<Scheme>,
    pub path_with_query: Option<String>,
    pub authority: Option<String>,
    pub headers: Fields,
    pub contents: Option<StreamReader<BytesMut>>,
    pub trailers: FutureReader<Result<Option<Resource<Fields>>, ErrorCode>>,
    pub options: Option<RequestOptions>,
}

pub struct Response {
    pub status_code: u16,
    pub headers: Fields,
    pub contents: Option<StreamReader<BytesMut>>,
    pub trailers: FutureReader<Result<Option<Resource<Fields>>, ErrorCode>>,
}

impl<T: WasiHttpView> wasi::http::types::HostFields for WasiHttpImpl<T> {
    fn new(&mut self) -> wasmtime::Result<Resource<Fields>> {
        Ok(self.table().push(Fields(Vec::new()))?)
    }

    fn from_list(
        &mut self,
        list: Vec<(String, Vec<u8>)>,
    ) -> wasmtime::Result<Result<Resource<Fields>, HeaderError>> {
        Ok(Ok(self.table().push(Fields(list))?))
    }

    fn get(&mut self, this: Resource<Fields>, key: String) -> wasmtime::Result<Vec<Vec<u8>>> {
        Ok(self
            .table()
            .get(&this)?
            .0
            .iter()
            .filter(|(k, _)| *k == key)
            .map(|(_, v)| v.clone())
            .collect())
    }

    fn has(&mut self, this: Resource<Fields>, key: String) -> wasmtime::Result<bool> {
        Ok(self.table().get(&this)?.0.iter().any(|(k, _)| *k == key))
    }

    fn set(
        &mut self,
        this: Resource<Fields>,
        key: String,
        values: Vec<Vec<u8>>,
    ) -> wasmtime::Result<Result<(), HeaderError>> {
        let fields = self.table().get_mut(&this)?;
        fields.0.retain(|(k, _)| *k != key);
        fields
            .0
            .extend(values.into_iter().map(|v| (key.clone(), v)));
        Ok(Ok(()))
    }

    fn get_and_delete(
        &mut self,
        this: Resource<Fields>,
        key: String,
    ) -> wasmtime::Result<Result<Vec<Vec<u8>>, HeaderError>> {
        let fields = self.table().get_mut(&this)?;
        let (matched, unmatched) = mem::take(&mut fields.0)
            .into_iter()
            .partition(|(k, _)| *k == key);
        fields.0 = unmatched;
        Ok(Ok(matched.into_iter().map(|(_, v)| v).collect()))
    }

    fn delete(
        &mut self,
        this: Resource<Fields>,
        key: String,
    ) -> wasmtime::Result<Result<(), HeaderError>> {
        self.get_and_delete(this, key).map(|v| v.map(drop))
    }

    fn append(
        &mut self,
        this: Resource<Fields>,
        key: String,
        value: Vec<u8>,
    ) -> wasmtime::Result<Result<(), HeaderError>> {
        self.table().get_mut(&this)?.0.push((key, value));
        Ok(Ok(()))
    }

    fn entries(&mut self, this: Resource<Fields>) -> wasmtime::Result<Vec<(String, Vec<u8>)>> {
        Ok(self.table().get(&this)?.0.clone())
    }

    fn clone(&mut self, this: Resource<Fields>) -> wasmtime::Result<Resource<Fields>> {
        let entries = self.table().get(&this)?.0.clone();
        Ok(self.table().push(Fields(entries))?)
    }

    fn drop(&mut self, this: Resource<Fields>) -> wasmtime::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl<C: WasiHttpViewConcurrent> wasi::http::types::HostRequestConcurrent for WasiHttp<C> {
    async fn new<T: 'static>(
        accessor: &Accessor<T, Self>,
        headers: Resource<Fields>,
        contents: Option<HostStream<u8>>,
        trailers: HostFuture<Result<Option<Resource<Fields>>, ErrorCode>>,
        options: Option<Resource<RequestOptions>>,
    ) -> wasmtime::Result<(Resource<Request>, HostFuture<Result<(), ErrorCode>>)> {
        accessor.with(|mut access| {
            let (_, result_rx) = access.instance().future(|| Ok(()), &mut access)?;

            let headers = access.get().table().delete(headers)?;
            let contents = contents.map(|v| v.into_reader(&mut access));
            let trailers = trailers.into_reader(&mut access);
            let options = if let Some(options) = options {
                Some(access.get().table().delete(options)?)
            } else {
                None
            };

            Ok((
                access.get().table().push(Request {
                    method: Method::Get,
                    scheme: None,
                    path_with_query: None,
                    authority: None,
                    headers,
                    contents,
                    trailers,
                    options,
                })?,
                result_rx.into(),
            ))
        })
    }
}

impl<T: WasiHttpView> wasi::http::types::HostRequest for WasiHttpImpl<T> {
    fn method(&mut self, this: Resource<Request>) -> wasmtime::Result<Method> {
        Ok(self.table().get(&this)?.method.clone())
    }

    fn set_method(
        &mut self,
        this: Resource<Request>,
        method: Method,
    ) -> wasmtime::Result<Result<(), ()>> {
        self.table().get_mut(&this)?.method = method;
        Ok(Ok(()))
    }

    fn scheme(&mut self, this: Resource<Request>) -> wasmtime::Result<Option<Scheme>> {
        Ok(self.table().get(&this)?.scheme.clone())
    }

    fn set_scheme(
        &mut self,
        this: Resource<Request>,
        scheme: Option<Scheme>,
    ) -> wasmtime::Result<Result<(), ()>> {
        self.table().get_mut(&this)?.scheme = scheme;
        Ok(Ok(()))
    }

    fn path_with_query(&mut self, this: Resource<Request>) -> wasmtime::Result<Option<String>> {
        Ok(self.table().get(&this)?.path_with_query.clone())
    }

    fn set_path_with_query(
        &mut self,
        this: Resource<Request>,
        path_with_query: Option<String>,
    ) -> wasmtime::Result<Result<(), ()>> {
        self.table().get_mut(&this)?.path_with_query = path_with_query;
        Ok(Ok(()))
    }

    fn authority(&mut self, this: Resource<Request>) -> wasmtime::Result<Option<String>> {
        Ok(self.table().get(&this)?.authority.clone())
    }

    fn set_authority(
        &mut self,
        this: Resource<Request>,
        authority: Option<String>,
    ) -> wasmtime::Result<Result<(), ()>> {
        self.table().get_mut(&this)?.authority = authority;
        Ok(Ok(()))
    }

    fn options(
        &mut self,
        this: Resource<Request>,
    ) -> wasmtime::Result<Option<Resource<RequestOptions>>> {
        // TODO: This should return an immutable child handle
        let options = self.table().get(&this)?.options;
        Ok(if let Some(options) = options {
            Some(self.table().push(options)?)
        } else {
            None
        })
    }

    fn headers(&mut self, this: Resource<Request>) -> wasmtime::Result<Resource<Fields>> {
        // TODO: This should return an immutable child handle
        let headers = self.table().get(&this)?.headers.clone();
        Ok(self.table().push(headers)?)
    }

    fn body(
        &mut self,
        _this: Resource<Request>,
    ) -> wasmtime::Result<
        Result<
            (
                HostStream<u8>,
                HostFuture<Result<Option<Resource<Fields>>, ErrorCode>>,
            ),
            (),
        >,
    > {
        Err(anyhow!("todo: implement `request.body`"))
    }

    fn drop(&mut self, this: Resource<Request>) -> wasmtime::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl<C: WasiHttpViewConcurrent> wasi::http::types::HostResponseConcurrent for WasiHttp<C> {
    async fn new<T: 'static>(
        accessor: &Accessor<T, Self>,
        headers: Resource<Fields>,
        contents: Option<HostStream<u8>>,
        trailers: HostFuture<Result<Option<Resource<Fields>>, ErrorCode>>,
    ) -> wasmtime::Result<(Resource<Response>, HostFuture<Result<(), ErrorCode>>)> {
        accessor.with(|mut access| {
            let (_, result_rx) = access.instance().future(|| Ok(()), &mut access)?;

            let headers = access.get().table().delete(headers)?;
            let contents = contents.map(|v| v.into_reader(&mut access));
            let trailers = trailers.into_reader(&mut access);

            Ok((
                access.get().table().push(Response {
                    status_code: 200,
                    headers,
                    contents,
                    trailers,
                })?,
                result_rx.into(),
            ))
        })
    }
}

impl<T: WasiHttpView> wasi::http::types::HostResponse for WasiHttpImpl<T> {
    fn status_code(&mut self, this: Resource<Response>) -> wasmtime::Result<u16> {
        Ok(self.table().get(&this)?.status_code)
    }

    fn set_status_code(
        &mut self,
        this: Resource<Response>,
        status_code: u16,
    ) -> wasmtime::Result<Result<(), ()>> {
        self.table().get_mut(&this)?.status_code = status_code;
        Ok(Ok(()))
    }

    fn headers(&mut self, this: Resource<Response>) -> wasmtime::Result<Resource<Fields>> {
        // TODO: This should return an immutable child handle
        let headers = self.table().get(&this)?.headers.clone();
        Ok(self.table().push(headers)?)
    }

    fn body(
        &mut self,
        _this: Resource<Response>,
    ) -> wasmtime::Result<
        Result<
            (
                HostStream<u8>,
                HostFuture<Result<Option<Resource<Fields>>, ErrorCode>>,
            ),
            (),
        >,
    > {
        Err(anyhow!("todo: implement `request.body`"))
    }

    fn drop(&mut self, this: Resource<Response>) -> wasmtime::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl<T: WasiHttpView> wasi::http::types::HostRequestOptions for WasiHttpImpl<T> {
    fn new(&mut self) -> wasmtime::Result<Resource<RequestOptions>> {
        Ok(self.table().push(RequestOptions::default())?)
    }

    fn connect_timeout(&mut self, this: Resource<RequestOptions>) -> wasmtime::Result<Option<u64>> {
        Ok(self.table().get(&this)?.connect_timeout)
    }

    fn set_connect_timeout(
        &mut self,
        this: Resource<RequestOptions>,
        connect_timeout: Option<u64>,
    ) -> wasmtime::Result<Result<(), RequestOptionsError>> {
        self.table().get_mut(&this)?.connect_timeout = connect_timeout;
        Ok(Ok(()))
    }

    fn first_byte_timeout(
        &mut self,
        this: Resource<RequestOptions>,
    ) -> wasmtime::Result<Option<u64>> {
        Ok(self.table().get(&this)?.first_byte_timeout)
    }

    fn set_first_byte_timeout(
        &mut self,
        this: Resource<RequestOptions>,
        first_byte_timeout: Option<u64>,
    ) -> wasmtime::Result<Result<(), RequestOptionsError>> {
        self.table().get_mut(&this)?.first_byte_timeout = first_byte_timeout;
        Ok(Ok(()))
    }

    fn between_bytes_timeout(
        &mut self,
        this: Resource<RequestOptions>,
    ) -> wasmtime::Result<Option<u64>> {
        Ok(self.table().get(&this)?.between_bytes_timeout)
    }

    fn set_between_bytes_timeout(
        &mut self,
        this: Resource<RequestOptions>,
        between_bytes_timeout: Option<u64>,
    ) -> wasmtime::Result<Result<(), RequestOptionsError>> {
        self.table().get_mut(&this)?.between_bytes_timeout = between_bytes_timeout;
        Ok(Ok(()))
    }

    fn clone(
        &mut self,
        this: Resource<RequestOptions>,
    ) -> wasmtime::Result<Resource<RequestOptions>> {
        let clone = *self.table().get(&this)?;
        Ok(self.table().push(clone)?)
    }

    fn drop(&mut self, this: Resource<RequestOptions>) -> wasmtime::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl<C: WasiHttpViewConcurrent> wasi::http::types::HostConcurrent for WasiHttp<C> {}

impl<T: WasiHttpView> wasi::http::types::Host for WasiHttpImpl<T> {}

impl<C: WasiHttpViewConcurrent> wasi::http::handler::HostConcurrent for WasiHttp<C> {
    async fn handle<T: 'static>(
        accessor: &Accessor<T, Self>,
        request: Resource<Request>,
    ) -> wasmtime::Result<Result<Resource<Response>, ErrorCode>> {
        SendRequestTask {
            request,
            _marker: marker::PhantomData,
        }
        .run(accessor)
        .await
    }
}

impl<T: WasiHttpView> wasi::http::handler::Host for WasiHttpImpl<T> {}

pub fn add_to_linker<T>(linker: &mut Linker<T>) -> wasmtime::Result<()>
where
    T: for<'a> WasiHttpViewConcurrent<View<'a> = &'a mut T> + 'static,
    T: WasiHttpView,
{
    wasi::http::types::add_to_linker::<T, WasiHttp<T>>(linker, |x| WasiHttpImpl(x))?;
    wasi::http::handler::add_to_linker::<T, WasiHttp<T>>(linker, |x| WasiHttpImpl(x))?;
    Ok(())
}
