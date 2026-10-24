//! iden3's Poseidon permutation of width 4 over the BN254 scalar field, after
//! <https://github.com/iden3/go-iden3-crypto/blob/e5cf066b8be3da9a3df9544c65818df189fdbebe/poseidon/poseidon.go>.

use ark_ff::Field;

use super::constants::CONSTANTS;
use super::Fr;

/// Field elements absorbed per permutation.
pub const RATE: usize = 3;
/// Field elements in the state.
pub const WIDTH: usize = 4;
pub const FULL_ROUNDS: usize = 8;
pub const PARTIAL_ROUNDS: usize = 56;

pub type PoseidonState = [Fr; WIDTH];

/// Applies the permutation in place. The partial rounds use the sparse
/// matrices of the optimized form (the S constants). Dot products go through
/// `sum_of_products`, which does one Montgomery reduction per chunk of
/// products instead of one per product.
pub fn permute(state: &mut PoseidonState) {
    let k = &*CONSTANTS;
    ark(state, &k.c[0..WIDTH]);
    for i in 0..FULL_ROUNDS / 2 - 1 {
        exp5_state(state);
        ark(state, &k.c[(i + 1) * WIDTH..(i + 2) * WIDTH]);
        mix(state, &k.m_t);
    }
    exp5_state(state);
    ark(
        state,
        &k.c[(FULL_ROUNDS / 2) * WIDTH..(FULL_ROUNDS / 2 + 1) * WIDTH],
    );
    mix(state, &k.p_t);

    for i in 0..PARTIAL_ROUNDS {
        let s0 = exp5(state[0]) + k.c[(FULL_ROUNDS / 2 + 1) * WIDTH + i];
        let s = &k.s[(WIDTH * 2 - 1) * i..(WIDTH * 2 - 1) * (i + 1)];
        let new_s0 = Fr::sum_of_products(
            &[s[0], s[1], s[2], s[3]],
            &[s0, state[1], state[2], state[3]],
        );
        for j in 1..WIDTH {
            state[j] += s0 * s[WIDTH + j - 1];
        }
        state[0] = new_s0;
    }

    for i in 0..FULL_ROUNDS / 2 - 1 {
        exp5_state(state);
        let at = (FULL_ROUNDS / 2 + 1) * WIDTH + PARTIAL_ROUNDS + i * WIDTH;
        ark(state, &k.c[at..at + WIDTH]);
        mix(state, &k.m_t);
    }
    exp5_state(state);
    mix(state, &k.m_t);
}

#[inline(always)]
fn ark(state: &mut PoseidonState, c: &[Fr]) {
    for (x, c) in state.iter_mut().zip(c) {
        *x += c;
    }
}

#[inline(always)]
fn exp5(x: Fr) -> Fr {
    x.square().square() * x
}

#[inline(always)]
fn exp5_state(state: &mut PoseidonState) {
    for x in state.iter_mut() {
        *x = exp5(*x);
    }
}

/// `state = m_t * state`, where `m_t` is the transposed MDS matrix.
#[inline(always)]
fn mix(state: &mut PoseidonState, m_t: &[[Fr; WIDTH]; WIDTH]) {
    let s = *state;
    for (x, row) in state.iter_mut().zip(m_t) {
        *x = Fr::sum_of_products(row, &s);
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use ark_ff::AdditiveGroup;

    use super::*;

    fn fr(s: &str) -> Fr {
        Fr::from_str(s).unwrap()
    }

    /// Outputs of iden3's implementation
    /// (<https://github.com/iden3/go-iden3-crypto/blob/master/poseidon/poseidon.go#L65>)
    /// for all zeros, `0..WIDTH`, all `p - 1` and an earlier output.
    #[test]
    fn iden3_vectors() {
        let max =
            fr("21888242871839275222246405745257275088548364400416034343698204186575808495616");
        let vectors: [([Fr; 4], [&str; 4]); 4] = [
            (
                [Fr::ZERO; 4],
                [
                    "5317387130258456662214331362918410991734007599705406860481038345552731150762",
                    "17768273200467269691696191901389126520069745877826494955630904743826040320364",
                    "19413739268543925182080121099097652227979760828059217876810647045303340666757",
                    "3717738800218482999400886888123026296874264026760636028937972004600663725187",
                ],
            ),
            (
                [fr("0"), fr("1"), fr("2"), fr("3")],
                [
                    "6542985608222806190361240322586112750744169038454362455181422643027100751666",
                    "3478427836468552423396868478117894008061261013954248157992395910462939736589",
                    "1904980799580062506738911865015687096398867595589699208837816975692422464009",
                    "11971464497515232077059236682405357499403220967704831154657374522418385384151",
                ],
            ),
            (
                [max; 4],
                [
                    "13055670547682322550638362580666986963569035646873545133474324633020685301274",
                    "19087936485076376314486368416882351797015004625427655501762827988254486144933",
                    "10391468779200270580383536396630001155994223659670674913170907401637624483385",
                    "17202557688472898583549180366140168198092766974201433936205272956998081177816",
                ],
            ),
            (
                [
                    fr("6542985608222806190361240322586112750744169038454362455181422643027100751666"),
                    fr("3478427836468552423396868478117894008061261013954248157992395910462939736589"),
                    fr("1904980799580062506738911865015687096398867595589699208837816975692422464009"),
                    fr("11971464497515232077059236682405357499403220967704831154657374522418385384151"),
                ],
                [
                    "21792249080447013894140672594027696524030291802493510986509431008224624594361",
                    "3536096706123550619294332177231935214243656967137545251021848527424156573335",
                    "14869351042206255711434675256184369368509719143073814271302931417334356905217",
                    "5027523131326906886284185656868809493297314443444919363729302983434650240523",
                ],
            ),
        ];
        for (mut state, expected) in vectors {
            permute(&mut state);
            assert_eq!(state, expected.map(fr));
        }
    }
}
