use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        //for name in safetensor.names() {
          //  println!("{}", name);
        //}
        let get_tensor= |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            // 直接转换整个切片
            let data = unsafe {
                let ptr = tensor.data().as_ptr() as *const f32;
                let len = tensor.data().len() / 4;
                std::slice::from_raw_parts(ptr, len).to_vec()
            };
            Tensor::<f32>::new(data, &tensor.shape().to_vec())
        };
        let num_layers = config.num_hidden_layers;
        let load_weight = |prefix: &str| -> Vec<Tensor<f32>>{
            (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}",i,prefix)))
                .collect()
        };
        LLamaParams{
            //story mode
            embedding_table: get_tensor(if config.tie_word_embeddings {"lm_head.weight"} else {"model.embed_tokens.weight"}),
            // decoder layer
            rms_att_w: load_weight("input_layernorm.weight"),
            wq: load_weight("self_attn.q_proj.weight"),
            wk: load_weight("self_attn.k_proj.weight"),
            wv: load_weight("self_attn.v_proj.weight"),
            wo: load_weight("self_attn.o_proj.weight"),
            // ffn layer
            rms_ffn_w: load_weight("post_attention_layernorm.weight"),
            w_up: load_weight("mlp.up_proj.weight"),
            w_gate: load_weight("mlp.gate_proj.weight"),
            w_down: load_weight("mlp.down_proj.weight"),
            // output
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head:get_tensor("lm_head.weight"),
        }
    }
}
