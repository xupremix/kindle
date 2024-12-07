use syn::{parse::Parse, Token};

// The macro can be called either with:
// just the dim (external use)
// shape!(3)
//
// or with the @ symbol to specify that it's an internal use

#[derive(Debug)]
pub(crate) enum AtTk {
    Present,
    Absent,
}

#[derive(Debug)]
pub(crate) struct Args {
    dim: i64,
    at_tk: AtTk,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let dim = input.parse::<syn::LitInt>()?.base10_parse()?;
        let at_tk = input
            .parse::<Token![@]>()
            .map_or_else(|_| AtTk::Absent, |_| AtTk::Present);

        Ok(Args { dim, at_tk })
    }
}
