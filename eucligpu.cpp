#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "OpenCLUtils.hpp"

#define KERNELNAME "euclidean"

class ExecuteOpenCL {
public:

  ExecuteOpenCL(const char *filename) {

    int imageWidth, imageHeight, imageChannels;
    image = stbi_load(filename, &imageWidth, &imageHeight, &imageChannels, 0);

    const int imageSize = imageWidth * imageHeight * imageChannels;
    output = new unsigned char[imageSize];
    executeOpenCL(KERNELNAME, ExecuteOpenCL::readKernel(), image, imageSize, output, imageSize);

    for (int i = 0; i < imageSize; ++i) {
      if (static_cast<int>(output[i]) != 2) {
        std::cout << "Verification failed! result #" << 2 << ", " << 
          static_cast<int>(output[i]) << " != 2 (expected)." << std::endl;

        throw std::exception();
      }
    }

    std::cout << "All right!" << std::endl;
  };

  ~ExecuteOpenCL() {

    if (image != nullptr)
      stbi_image_free(image);
    if (output != nullptr)
      free(output);

  };

private:

  unsigned char *image;
  unsigned char *output;

  static std::string readKernel() {
    std::ifstream input("kernel.cl");
    std::string source;
    input.seekg(0, std::ios::end);
    source.reserve(input.tellg());
    input.seekg(0, std::ios::beg);
    source.assign((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return source;
  }

};


int main(int argc, char const *argv[]) {
  if (argc < 2) {
      std::cerr << "You have to pass 1 argument to the program, but none was passed." << std::endl;
      return -1;
  }

  const char *filename = argv[1];

  // Executa com o destrutor seguro para desalocar todos os ponteiros criados.
  ExecuteOpenCL exec(filename);

  return 0;
}
