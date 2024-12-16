use candle_nn::Sequential;

use crate::prelude::Vs;

use super::ModelBuilder;

macro_rules! tuple_model {
    ($( $($M:ident)* - $($c:ident)* );* $(;)?) => {
        $(
            impl<$($M: ModelBuilder),*> ModelBuilder for ($($M),*) {
                type Config = ($($M::Config),*);
                fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
                    let ($($c),*) = c;
                    $(let seq = $M::step(vs, $c, seq);)*
                    seq
                }
            }
        )*
    };
}

tuple_model! {
    M0 M1 - c0 c1;
    M0 M1 M2 - c0 c1 c2;
    M0 M1 M2 M3 - c0 c1 c2 c3;
    M0 M1 M2 M3 M4 - c0 c1 c2 c3 c4;
    M0 M1 M2 M3 M4 M5 - c0 c1 c2 c3 c4 c5;
    M0 M1 M2 M3 M4 M5 M6 - c0 c1 c2 c3 c4 c5 c6;
    M0 M1 M2 M3 M4 M5 M6 M7 - c0 c1 c2 c3 c4 c5 c6 c7;
    M0 M1 M2 M3 M4 M5 M6 M7 M8 - c0 c1 c2 c3 c4 c5 c6 c7 c8;
    M0 M1 M2 M3 M4 M5 M6 M7 M8 M9 - c0 c1 c2 c3 c4 c5 c6 c7 c8 c9;
}
