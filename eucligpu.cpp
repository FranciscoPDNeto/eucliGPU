#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "OpenCLUtils.hpp"

#define KERNELNAME "euclidean"
// TODO colocar o kernel como requerido
#define KERNELSRC "void kernel " KERNELNAME "(global const unsigned char* A, " \
  "  global unsigned char* B) {\n"                                           \
  "\n"                                                                       \
  "  unsigned long taskIndex = get_global_id(0);\n"                          \
  "  B[taskIndex] = A[taskIndex] + A[taskIndex];\n"                          \
  "}"

class ExecuteOpenCL {
public:

  ExecuteOpenCL(const std::string& filename) : m_filename(filename), m_image(nullptr), m_output(nullptr) {
  };

  void execute() {
    int imageWidth, imageHeight, imageChannels;
    m_image = stbi_load(m_filename.c_str(), &imageWidth, &imageHeight, &imageChannels, 0);
    if (m_image == nullptr)
      throw std::runtime_error(
        "The image could not be loaded, please check if the filename is corrected");

    const int imageSize = imageWidth * imageHeight * imageChannels;
    m_output = new unsigned char[imageSize];
    assert(m_output != nullptr);

    executeOpenCL(KERNELNAME, KERNELSRC, m_image, imageSize, m_output, imageSize);

    for (int i = 0; i < imageSize; ++i) {
      if (static_cast<int>(m_output[i]) != 2) {
        throw std::runtime_error("Verification failed! result #2 , " +
          std::to_string(static_cast<int>(m_output[i])) + " != 2 (expected).");
      }
    }

    std::cout << "All right!" << std::endl;
  }

  ~ExecuteOpenCL() {
    if (m_image != nullptr)
      stbi_image_free(m_image);
    
    if (m_output != nullptr)
      free(m_output);

  };

private:
  const std::string m_filename;
  unsigned char *m_image;
  unsigned char *m_output;
};


int main(int argc, char const *argv[]) {
  if (argc < 2) {
      std::cerr << "You have to pass 1 argument to the program, but none was passed." << std::endl;
      return -1;
  }

  const std::string filename(argv[1]);
  
  try {
    // Executa com o destrutor seguro para desalocar todos os ponteiros criados.
    ExecuteOpenCL exec(filename);
    exec.execute();
  } catch (const std::runtime_error& e) {
    throw e;
  }

  return 0;
}
