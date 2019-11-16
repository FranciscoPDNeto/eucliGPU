#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "OpenCLUtils.hpp"

#define KERNELNAME "euclidean"
// TODO right kernel source
#define KERNELSRC "void kernel " KERNELNAME "(global const unsigned char* A, " \
  "  global unsigned char* B) {\n"                                           \
  "\n"                                                                       \
  "  unsigned long taskIndex = get_global_id(0);\n"                          \
  "  B[taskIndex] = A[taskIndex] + A[taskIndex];\n"                          \
  "}"


int main(int argc, char const *argv[]) {
  /*
  if (argc < 2) {
      std::cerr << "You have to pass 1 argument to the program, but none was passed.";
      return -1;
  }

  const char *filename = argv[1];

  int imageWidth, imageHeight, imageChannels;
  unsigned char *image = stbi_load(filename, &imageWidth, &imageHeight, &imageChannels, 0);

  const int imageSize = imageWidth * imageHeight;
  unsigned char *output = new unsigned char[imageSize];
  executeOpenCL(KERNELNAME, KERNELSRC, image, imageSize, output, imageSize);
  stbi_image_free(image);
  */

  const unsigned char test[] = {1,1,1,1,1,1,1,1,1,1};
  unsigned char *output = new unsigned char[10];

  executeOpenCL(KERNELNAME, KERNELSRC, test, 10, output, 10);

  for (std::size_t i = 0; i < 10; ++i) {
    if (static_cast<int>(output[i]) != 2) {
      std::cout << "Verification failed! result #" << 1 << ", " << 
        output[i] << " != " << i << " (expected)." << std::endl;

      return -1;
    }
  }

  std::cout << "All right!" << std::endl;

  return 0;
}
