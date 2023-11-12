#include <torch/extension.h>

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *last_state, float *y,  float *new_state);
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *last_state, float *y, float *gy, float *gnew_state, float *gw, float *gu, float *gk, float *gv, float *glast_state);

void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &last_state, torch::Tensor &y, torch::Tensor &new_state) {
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), last_state.data_ptr<float>(), y.data_ptr<float>(),  new_state.data_ptr<float>());
}
void backward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &last_state, 
    torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gnew_state, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &glast_state) {
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), last_state.data_ptr<float>(),
    y.data_ptr<float>(), gy.data_ptr<float>(), gnew_state.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>(), glast_state.data_ptr<float>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
