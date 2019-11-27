#include <iostream>
#include <stdexcept>

#include "ImageUtils.hpp"

namespace OpenCLUtils {

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

std::vector<cl::Device> getDevices() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.size() == 0) {
    throw std::runtime_error(
        "There is no platforms available. Check OpenCL installation!");
  }

  cl::Platform defaultPlatform = platforms[1];
  std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>()
            << "\n";

  std::vector<cl::Device> devices;
  defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.size() == 0) {
    throw std::runtime_error(
        "There is no devices available. Check OpenCL installation!");
  }

  return devices;
}

cl::Device getDevice(int deviceId) {
  std::vector<cl::Device> devices(getDevices());

  if (deviceId > static_cast<int>(devices.size())) {
    throw std::runtime_error("There is just " + std::to_string(devices.size()) +
                             " devices available, so there is no device id " +
                             std::to_string(deviceId));
  }

  const cl::Device device = devices[deviceId];
  std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
  return device;
}

cl::Device userSelectDevice() {
  std::vector<cl::Device> devices(getDevices());

  for (std::size_t i = 0; i < devices.size(); ++i) {
    std::cout << "Device " << i << " - " << devices[i].getInfo<CL_DEVICE_NAME>()
              << std::endl;
  }

  std::cout << "Select your device id: ";
  int deviceId;
  std::cin >> deviceId;
  cl::Device defaultDevice = devices[deviceId];
  std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>()
            << "\n";

  return defaultDevice;
}

static std::string readKernel() {
  std::ifstream input("kernel2.cl");
  std::string source;
  input.seekg(0, std::ios::end);
  source.reserve(input.tellg());
  input.seekg(0, std::ios::beg);
  source.assign((std::istreambuf_iterator<char>(input)),
                std::istreambuf_iterator<char>());
  return source;
}

void executeAlternativeKernel(const UCImage *image, cl_float *dataOutput) {
  const cl::Device defaultDevice = getDevice(0);

  cl::Context context({defaultDevice});
  cl::Program::Sources sources;

  const std::string kernelSource(readKernel());
  sources.push_back({kernelSource.c_str(), kernelSource.length()});
  cl::Program program(context, sources);
  if (program.build({defaultDevice}) != CL_SUCCESS) {
    throw std::runtime_error(
        "Error building: " +
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice));
  }

  cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(cl_uchar)*image->attrs.size, nullptr);
  cl::Buffer inputAttrsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(ImageAttrs), nullptr);
  cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(cl_float)*image->attrs.size, nullptr);

  cl::CommandQueue queue(context, defaultDevice);
  
  cl_int errorCode = queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0,
                           sizeof(cl_uchar)*image->attrs.size, image->image);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  errorCode = queue.enqueueWriteBuffer(inputAttrsBuffer, CL_TRUE, 0,
                           sizeof(ImageAttrs), &(image->attrs));
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  errorCode = queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0,
                           sizeof(cl_float)*image->attrs.size, dataOutput);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));
  
  const int localSize = 32;
  cl::Kernel kernel(program, "DistanceTransform");
  kernel.setArg(0, inputBuffer);
  kernel.setArg(1, inputAttrsBuffer);
  kernel.setArg(2, outputBuffer);

  errorCode = queue.enqueueNDRangeKernel(kernel, 0, image->attrs.size, localSize);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  // Retorna o resultado da computação na GPU para o dataOutput.
  errorCode = queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0,
                          sizeof(cl_float) * image->attrs.size, dataOutput);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));
}

void executeOpenCL(const std::string &kernelName,
                   const std::string &kernelSource,
                   const UCImage *image,
                   const std::vector<cl_uint4>& pixelQueue,
                   const VoronoiDiagramMap *voronoi,
                   cl_float *dataOutput) {

  const cl::Device defaultDevice = getDevice(0);

  cl::Context context({defaultDevice});
  cl::Program::Sources sources;

  sources.push_back({kernelSource.c_str(), kernelSource.length()});
  cl::Program program(context, sources);
  if (program.build({defaultDevice}) != CL_SUCCESS) {
    throw std::runtime_error(
        "Error building: " +
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice));
  }

  cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(cl_uchar)*image->attrs.size, nullptr);
  cl::Buffer inputAttrsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(ImageAttrs), nullptr);
  cl::Buffer inputQueueBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(cl_uint4)*pixelQueue.size(), nullptr);
  cl::Buffer outputVoronoiBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(VoronoiDiagramMapEntry)*voronoi->sizeOfDiagram, nullptr);

  cl::CommandQueue queue(context, defaultDevice);
  
  cl_int errorCode = queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0,
                           sizeof(cl_uchar)*image->attrs.size, image->image);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  errorCode = queue.enqueueWriteBuffer(inputAttrsBuffer, CL_TRUE, 0,
                           sizeof(ImageAttrs), &(image->attrs));
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  errorCode = queue.enqueueWriteBuffer(inputQueueBuffer, CL_FALSE, 0,
                           sizeof(cl_uint4)*pixelQueue.size(), pixelQueue.data());
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  errorCode = queue.enqueueWriteBuffer(outputVoronoiBuffer, CL_TRUE, 0,
                           sizeof(VoronoiDiagramMapEntry)*voronoi->sizeOfDiagram, voronoi->entries);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));
  
  const int localSize = 32;
  const unsigned int pixelQueueSize = pixelQueue.size();
  cl::Kernel kernel(program, kernelName.c_str());
  kernel.setArg(0, inputBuffer);
  kernel.setArg(1, inputAttrsBuffer);
  kernel.setArg(2, inputQueueBuffer);
  kernel.setArg(3, sizeof(unsigned int), &pixelQueueSize);
  kernel.setArg(4, cl::Local(pixelQueue.size()/localSize + pixelQueue.size()%localSize));
  kernel.setArg(5, outputVoronoiBuffer);
  kernel.setArg(6, sizeof(unsigned int), &voronoi->sizeOfDiagram);

  errorCode = queue.enqueueNDRangeKernel(kernel, 0, image->attrs.size, localSize);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));

  // Retorna o resultado da computação na GPU para o dataOutput.
  errorCode = queue.enqueueReadBuffer(outputVoronoiBuffer, CL_TRUE, 0,
                          sizeof(VoronoiDiagramMapEntry) * voronoi->sizeOfDiagram, voronoi->entries);
  if (errorCode != CL_SUCCESS)
    throw std::runtime_error(getErrorString(errorCode));
}

} // namespace OpenCLUtils