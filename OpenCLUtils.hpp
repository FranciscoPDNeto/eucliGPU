#include <iostream>
#include <stdexcept>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

namespace OpenCLUtils {

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

void executeOpenCL(const std::string &kernelName,
                   const std::string &kernelSource,
                   const unsigned char *dataInput, const int dataInputSize,
                   float *dataOutput, const int dataOutputSize) {

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

  // Estudando alterar para Image2D
  /*
  cl::ImageFormat format(CL_INTENSITY, CL_UNORM_INT8);
  cl::Image2D image(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, format, imageWidth, 
    imageHeight, imageWidth * sizeof(unsigned char), image);
  */
  cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         sizeof(unsigned char) * dataInputSize, nullptr);
  cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          sizeof(float) * dataOutputSize, nullptr);

  cl::CommandQueue queue(context, defaultDevice);
  queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0,
                           sizeof(unsigned char) * dataInputSize, dataInput);

  cl::Kernel kernel(program, kernelName.c_str());
  kernel.setArg(0, inputBuffer);
  kernel.setArg(1, outputBuffer);

  queue.enqueueNDRangeKernel(kernel, 0, dataInputSize, 32);

  // Retorna o resultado da computação na GPU para o dataOutput.
  queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0,
                          sizeof(float) * dataOutputSize, dataOutput);
}

} // namespace OpenCLUtils