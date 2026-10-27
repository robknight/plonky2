//! Experimental device-side ChaCha20 with rejection sampling into Goldilocks.
//! Each call receives a fresh secret key; domains separate witness and salts.
use rand::RngCore;

use super::{DeviceBuffer, GpuContext};

pub(crate) fn fresh_seed() -> zeroize::Zeroizing<[u8; 32]> {
    let mut seed = zeroize::Zeroizing::new([0u8; 32]);
    rand::rngs::OsRng.fill_bytes(seed.as_mut());
    seed
}
pub(crate) fn seed_words(seed: &[u8; 32], domain: u32) -> zeroize::Zeroizing<Vec<u32>> {
    let mut words = seed
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| u32::from_le_bytes(*x))
        .collect::<Vec<_>>();
    words.extend([domain, 0, 0, 0]);
    zeroize::Zeroizing::new(words)
}
pub(crate) struct Kernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    pub(crate) fn fill_random(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        count: usize,
        seed: &[u8; 32],
        domain: u32,
    ) {
        if count == 0 {
            return;
        }
        let k = self.random.get_or_init(|| {
            let layout = super::storage_layout(
                &self.device,
                "device randomness",
                &[(true, true), (true, true), (false, false)],
            );
            let source = format!("{CHACHA}\n{FILL}");
            let module = super::module(&self.device, "device ChaCha20", &source);
            Kernels {
                pipeline: super::pipeline(&self.device, &layout, &module, "fill"),
                layout,
            }
        });
        let key = self.uniform("fresh random key", &seed_words(seed, domain));
        let params = self.uniform("random range", &[offset as u32, count as u32, 0, 0]);
        let bind = self.bind(&k.layout, &[&params, &key, dst]);
        let mut encoder = self.encoder("device randomness");
        {
            let mut pass = self.begin_pass(&mut encoder, "device randomness", None);
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (count as u32).div_ceil(8).div_ceil(128);
            let width = groups.min(65535);
            pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
        }
        self.submit_async(encoder);
    }
}
#[cfg(test)]
pub(crate) fn reference(seed: [u8; 32], domain: u32, index: usize) -> u64 {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(seed);
    let mut attempt = 0u64;
    loop {
        rng.set_stream((attempt << 32) | domain as u64);
        rng.set_word_pos((index / 8 * 16 + index % 8 * 2) as u128);
        let value = rng.next_u64();
        if value < 0xffff_ffff_0000_0001 {
            return value;
        }
        attempt += 1;
    }
}

pub(crate) const CHACHA: &str = r#"
struct RandomKey {a:vec4<u32>,b:vec4<u32>,domain:vec4<u32>}
fn rol(x:u32,n:u32)->u32 {return (x<<n)|(x>>(32u-n));}
fn qr(p:ptr<function,array<u32,16>>,a:u32,b:u32,c:u32,d:u32) {
 (*p)[a]+=(*p)[b];(*p)[d]=rol((*p)[d]^(*p)[a],16u);
 (*p)[c]+=(*p)[d];(*p)[b]=rol((*p)[b]^(*p)[c],12u);
 (*p)[a]+=(*p)[b];(*p)[d]=rol((*p)[d]^(*p)[a],8u);
 (*p)[c]+=(*p)[d];(*p)[b]=rol((*p)[b]^(*p)[c],7u);
}
fn chacha(key:RandomKey,block:u32,attempt:u32)->array<u32,16> {
 let initial=array<u32,16>(0x61707865u,0x3320646eu,0x79622d32u,0x6b206574u,
 key.a.x,key.a.y,key.a.z,key.a.w,key.b.x,key.b.y,key.b.z,key.b.w,
 block,0u,key.domain.x,attempt);
 var p=initial;
 for(var r=0u;r<10u;r++) {
  qr(&p,0u,4u,8u,12u);qr(&p,1u,5u,9u,13u);qr(&p,2u,6u,10u,14u);qr(&p,3u,7u,11u,15u);
  qr(&p,0u,5u,10u,15u);qr(&p,1u,6u,11u,12u);qr(&p,2u,7u,8u,13u);qr(&p,3u,4u,9u,14u);
 }
 for(var i=0u;i<16u;i++){p[i]+=initial[i];}return p;
}
fn accepted(v:vec2<u32>)->bool {return v.y!=0xffffffffu || v.x==0u;}
fn random_field(key:RandomKey,index:u32)->vec2<u32> {
 var attempt=0u;
 loop {
  let words=chacha(key,index/8u,attempt);let pos=(index%8u)*2u;
  let value=vec2<u32>(words[pos],words[pos+1u]);
  if(accepted(value)){return value;}attempt++;
 }
 return vec2<u32>(0u);
}
"#;
const FILL: &str = r#"
struct Range {offset:u32,count:u32,pad0:u32,pad1:u32}
@group(0) @binding(0) var<uniform> range:Range;
@group(0) @binding(1) var<uniform> key:RandomKey;
@group(0) @binding(2) var<storage,read_write> output:array<vec2<u32>>;
@compute @workgroup_size(128)
fn fill(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let block=id.x+id.y*groups.x*128u;let begin=block*8u;if(begin>=range.count){return;}
 let words=chacha(key,block,0u);
 for(var i=0u;i<8u && begin+i<range.count;i++) {
  var value=vec2<u32>(words[2u*i],words[2u*i+1u]);
  if(!accepted(value)){value=random_field(key,begin+i);}
  output[range.offset+begin+i]=value;
 }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires native GPU; run alone"]
    fn device_random_matches_chacha20_and_preserves_ranges() {
        let ctx = super::super::context().expect("native GPU");
        let seed = core::array::from_fn(|i| i as u8);
        for domain in [0, 0x5749544e, 0x53414c54] {
            let dst = ctx.upload("random test sentinels", &vec![77; 1040]);
            ctx.fill_random(&dst, 3, 1027, &seed, domain);
            let actual = ctx.read_u64(&dst, 1040);
            assert_eq!(&actual[..3], &[77; 3]);
            assert_eq!(&actual[1030..], &[77; 10]);
            for (i, &value) in actual[3..1030].iter().enumerate() {
                assert_eq!(value, reference(seed, domain, i), "domain={domain} i={i}");
            }
        }
        let a = fresh_seed();
        let b = fresh_seed();
        assert_ne!(a.as_ref(), b.as_ref());
    }
}
