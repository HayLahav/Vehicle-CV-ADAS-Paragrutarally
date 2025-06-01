import abc
import os
import numpy as np
import onnxruntime
import platform

# Check if running on Jetson
def is_jetson():
    """Check if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release') as f:
            return True
    except:
        return False

# Import TensorRT only if available
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    print("TensorRT not available, falling back to ONNX Runtime")

class EngineBase(abc.ABC):
    '''
    Currently supports Onnx/TensorRT framework with Jetson compatibility
    '''
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The model path [%s] can't not found!" % model_path)
        assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'
        self._framework_type = None

    @property
    def framework_type(self):
        if (self._framework_type == None):
            raise Exception("Framework type can't be None")
        return self._framework_type
    
    @framework_type.setter
    def framework_type(self, value):
        if ( not isinstance(value, str)):
            raise Exception("Framework type need be str")
        self._framework_type = value
    
    @abc.abstractmethod
    def get_engine_input_shape(self):
        return NotImplemented
    
    @abc.abstractmethod
    def get_engine_output_shape(self):
        return NotImplemented
    
    @abc.abstractmethod
    def engine_inference(self):
        return NotImplemented

class TensorRTEngine(EngineBase):
    """Jetson-compatible TensorRT Engine without PyCUDA dependency"""
    
    def __init__(self, engine_file_path):
        EngineBase.__init__(self, engine_file_path)
        
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install TensorRT or use ONNX models.")
        
        self.framework_type = "trt"
        self.providers = 'TensorRT'
        
        # Initialize TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        self._load_engine(engine_file_path)
        
        # Setup bindings
        self._setup_bindings()
        
    def _load_engine(self, engine_file_path):
        """Load TensorRT engine from file"""
        with open(engine_file_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_file_path}")
        
        self.context = self.engine.create_execution_context()
        
    def _setup_bindings(self):
        """Setup input/output bindings for the engine"""
        self.inputs = []
        self.outputs = []
        self.input_shapes = []
        self.output_shapes = []
        self.output_names = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'index': i
                })
                self.input_shapes.append(shape)
            else:
                self.outputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'index': i
                })
                self.output_shapes.append(shape)
                self.output_names.append(name)
        
        self.engine_dtype = self.inputs[0]['dtype']
    
    def get_engine_input_shape(self):
        return self.input_shapes[0] if self.input_shapes else None
    
    def get_engine_output_shape(self):
        return self.output_shapes, self.output_names
    
    def engine_inference(self, input_tensor):
        """Run inference using TensorRT"""
        try:
            # Ensure input tensor is contiguous and correct dtype
            if not input_tensor.flags['C_CONTIGUOUS']:
                input_tensor = np.ascontiguousarray(input_tensor)
            
            if input_tensor.dtype != self.engine_dtype:
                input_tensor = input_tensor.astype(self.engine_dtype)
            
            # Set input shape (for dynamic shapes)
            if self.inputs[0]['shape'][0] == -1:  # Dynamic batch size
                input_shape = input_tensor.shape
                self.context.set_binding_shape(0, input_shape)
            
            # Allocate memory for outputs
            outputs = []
            bindings = [None] * self.engine.num_bindings
            
            # Set input binding
            bindings[self.inputs[0]['index']] = input_tensor.ctypes.data_as(trt.ctypes.c_void_p)
            
            # Allocate output buffers
            for output in self.outputs:
                output_shape = self.context.get_binding_shape(output['index'])
                output_buffer = np.empty(output_shape, dtype=output['dtype'])
                outputs.append(output_buffer)
                bindings[output['index']] = output_buffer.ctypes.data_as(trt.ctypes.c_void_p)
            
            # Execute inference
            success = self.context.execute_v2(bindings)
            
            if not success:
                raise RuntimeError("TensorRT inference failed")
            
            return outputs
            
        except Exception as e:
            raise RuntimeError(f"TensorRT inference error: {str(e)}")

class JetsonOnnxEngine(EngineBase):
    """ONNX Runtime with TensorRT execution provider for Jetson"""
    
    def __init__(self, onnx_file_path):
        EngineBase.__init__(self, onnx_file_path)
        
        # Configure providers for Jetson
        if is_jetson():
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 1 << 30,  # 1GB
                    'trt_fp16_enable': True,
                    'trt_max_partition_iterations': 1000,
                    'trt_min_subgraph_size': 1,
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                }),
                'CPUExecutionProvider'
            ]
        else:
            # Standard CUDA setup for non-Jetson platforms
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                }),
                'CPUExecutionProvider'
            ]
        
        try:
            self.session = onnxruntime.InferenceSession(onnx_file_path, providers=providers)
        except Exception as e:
            print(f"Failed to create session with GPU providers, falling back to CPU: {e}")
            self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
        
        self.providers = self.session.get_providers()
        self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        self.framework_type = "onnx-trt" if is_jetson() else "onnx"
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = [input.shape for input in self.session.get_inputs()]
        self.__input_names = [input.name for input in self.session.get_inputs()]
        self.__output_shape = [output.shape for output in self.session.get_outputs()]
        self.__output_names = [output.name for output in self.session.get_outputs()]

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shape, self.__output_names
    
    def engine_inference(self, input_tensor):
        output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
        return output

class OnnxEngine(EngineBase):
    """Standard ONNX Runtime Engine"""

    def __init__(self, onnx_file_path):
        EngineBase.__init__(self, onnx_file_path)
        
        # Choose providers based on availability
        available_providers = onnxruntime.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        try:
            self.session = onnxruntime.InferenceSession(onnx_file_path, providers=providers)
        except Exception as e:
            print(f"Failed to create ONNX session with preferred providers, using default: {e}")
            self.session = onnxruntime.InferenceSession(onnx_file_path)
        
        self.providers = self.session.get_providers()
        self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        self.framework_type = "onnx"
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = [input.shape for input in self.session.get_inputs()]
        self.__input_names = [input.name for input in self.session.get_inputs()]
        self.__output_shape = [output.shape for output in self.session.get_outputs()]
        self.__output_names = [output.name for output in self.session.get_outputs()]

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shape, self.__output_names
    
    def engine_inference(self, input_tensor):
        output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
        return output

# Factory function to choose the best engine
def create_engine(model_path):
    """Factory function to create the most appropriate engine"""
    if model_path.endswith('.trt'):
        if TRT_AVAILABLE and is_jetson():
            try:
                return TensorRTEngine(model_path)
            except Exception as e:
                print(f"TensorRT engine creation failed: {e}")
                # Try to find corresponding ONNX file
                onnx_path = model_path.replace('.trt', '.onnx')
                if os.path.exists(onnx_path):
                    print(f"Falling back to ONNX: {onnx_path}")
                    return JetsonOnnxEngine(onnx_path) if is_jetson() else OnnxEngine(onnx_path)
                else:
                    raise Exception(f"No fallback ONNX model found for {model_path}")
        else:
            raise Exception("TensorRT not available or not on Jetson platform")
    else:
        # ONNX model
        if is_jetson():
            return JetsonOnnxEngine(model_path)
        else:
            return OnnxEngine(model_path)