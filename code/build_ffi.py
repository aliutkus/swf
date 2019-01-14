# https://gist.github.com/tonyseek/7821993
import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))

# Preparing the info for the CPU version
sources = ['src_searchsorted/src_cpu/searchsorted_cpu_wrapper.c', ]
headers = ['src_searchsorted/src_cpu/searchsorted_cpu_wrapper.h']
extra_objects = []
define_macros = []
include_dirs = [osp.join(abs_path, 'src_searchsorted/src_cpu')]

cuda_available = torch.cuda.is_available()
if cuda_available:
    # if available, adding the info for the CUDA version
    extra_objects = [osp.join(
            abs_path,
            'build/searchsorted_cuda_kernel.so')]
    extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')
    define_macros = [('WITH_CUDA', None)]
    sources += ['src_searchsorted/src_cuda/searchsorted_cuda_wrapper.c']
    headers += ['src_searchsorted/src_cuda/searchsorted_cuda_wrapper.h']
    include_dirs = [osp.join(abs_path, 'src_searchsorted/src_cuda')]

ffi = create_extension(
    'searchsorted.searchsorted_wrapper',
    headers=headers,
    sources=sources,
    define_macros=define_macros,
    relative_to=__file__,
    with_cuda=cuda_available,
    extra_objects=extra_objects,
    include_dirs=include_dirs
)

if __name__ == '__main__':
    ffi.build()
