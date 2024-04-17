// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rustacuda::memory::DeviceCopy;


// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}


pub struct OutputThreadVect(pub [[f64; 32]; OUT_LAYER_SIZE]);
unsafe impl DeviceCopy for OutputThreadVect{}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        // Initialize API
        rustacuda::init(CudaFlags::empty())?;

        // Get Device and set context
        let device = Device::get_device(0);
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device?)?;

        // Load Module and create stream
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        Ok(Self {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            _context: _ctx,
            module,
            stream
        })

    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        const THREAD_NUM: usize = 32;

        let mut input = DeviceBox::new(input)?;
        let mut conv_kernel_output = DeviceBox::new(&[[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE])?;

        let mut result = OutputVec([0.0; OUT_LAYER_SIZE]);
        let mut output_holder = OutputThreadVect([[0.0; THREAD_NUM]; OUT_LAYER_SIZE]);
        let mut output_device = DeviceBox::new(&output_holder)?;

        let conv_kernel_grid = GridSize::x(OUT_LAYER_SIZE as u32);
        let conv_kernel_block = BlockSize::xy(CONV_OUT_DIM as u32, CONV_OUT_DIM as u32);
        let output_grid = GridSize::x(OUT_LAYER_SIZE as u32);
        let output_block = BlockSize::x(THREAD_NUM as u32);

        let module = &self.module;
        let stream = &self.stream;

        // Need to launch in unsafe {} as we can no longer gurantee the code will hold - CUDA
        unsafe {
            let _ = launch!(module.convolution_relu_kernel<<<conv_kernel_grid, conv_kernel_block, 0, stream>>>(
                self.conv_layer.as_device_ptr(),
                input.as_device_ptr(),
                conv_kernel_output.as_device_ptr()
            ));

            let _ = launch!(module.output_kernel<<<output_grid, output_block, 0, stream>>>(
                conv_kernel_output.as_device_ptr(),
                output_device.as_device_ptr(),
                self.output_layer.as_device_ptr()
            ));
        }

        let _ = stream.synchronize()?;

        output_device.copy_to(&mut output_holder)?;

        for x in 0..OUT_LAYER_SIZE {
            for y in 0..THREAD_NUM {
                result.0[x] += output_holder.0[x][y];
            }
        }

        Ok(result)
        

    }
}
