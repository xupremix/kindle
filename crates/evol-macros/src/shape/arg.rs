use syn::{parse::Parse, Token, Visibility};

// The macro can be called either with:
// just the dim (external use)
// shape!(3)
//
// or with the @ symbol to specify that it's an internal use

#[derive(Debug)]
pub(crate) struct Args {
    pub(crate) vis: Visibility,
    pub(crate) dims: usize,
    pub(crate) at_tk: bool,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let vis = input.parse()?;
        let dims = input.parse::<syn::LitInt>()?.base10_parse()?;
        let at_tk = input.parse::<Token![@]>().map_or_else(|_| false, |_| true);
        Ok(Args { vis, dims, at_tk })
    }
}
