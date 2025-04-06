use syn::{parse::Parse, Ident, LitStr, Token};

#[derive(Debug)]
pub(crate) struct Args {
    pub(crate) name: Ident,
    pub(crate) path: String,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let path: LitStr = input.parse()?;
        Ok(Args {
            name,
            path: path.value(),
        })
    }
}
