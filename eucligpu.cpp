#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OpenCLUtils.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#define KERNELNAME "euclidean"

void computeDistanceTransformImage(const unsigned char *imageInput, float *imageOutput,
  const int imageWidth, const int imageHeight) {
  static const float thresholdVal = 1.0f;

  for(int x=0; x < imageWidth; ++x) {
    for(int y=0; y < imageHeight; ++y) {

      float minDistance = std::numeric_limits<float>::max();

      for(int ox=0; ox < imageWidth; ++ox) {
        for(int oy=0; oy < imageHeight; ++oy) {

          if(static_cast<float>(imageInput[imageHeight*oy + ox]) >= thresholdVal) {
            const float distance = std::sqrt( 
              static_cast<float>((ox-x)*(ox-x)) + 
              static_cast<float>((oy-y)*(oy-y)) );

            if( distance < minDistance )
              minDistance = distance;
            
          }
        }
      }
      imageOutput[imageHeight*y + x] = minDistance;
    }
  }

}


class ExecuteOpenCL {
public:
  ExecuteOpenCL(const std::string &filename)
      : m_filename(filename), m_image(nullptr), m_output(nullptr){};

  void execute() {
    // As imagens esperadas são sempre com apenas um canal.
    int imageWidth, imageHeight;
    m_image = stbi_load(m_filename.c_str(), &imageWidth, &imageHeight,
                        nullptr, 1);
    if (m_image == nullptr)
      throw std::runtime_error("The image could not be loaded, please check if "
                               "the filename is corrected");

    const int imageSize = imageWidth * imageHeight;
    m_output = new float[imageSize];
    assert(m_output != nullptr);

    // Paralelo
    OpenCLUtils::executeOpenCL(KERNELNAME, ExecuteOpenCL::readKernel(), m_image,
                               imageSize, m_output, imageSize);

    // Sequencial
    //computeDistanceTransformImage(m_image, m_output, imageWidth, imageHeight);
    for (int i = 0; i < imageSize; i++)
      printf("%f ", m_output[i]);
    printf("\n");
    // A escrita na imagem de saída ainda deve ser consertada, esse método sempre
    // espera um buffer de unsigned char...
    stbi_write_bmp("result.bmp", imageWidth, imageHeight, 1, m_output);
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
  float *m_output;

  static std::string readKernel() {
    std::ifstream input("kernel.cl");
    std::string source;
    input.seekg(0, std::ios::end);
    source.reserve(input.tellg());
    input.seekg(0, std::ios::beg);
    source.assign((std::istreambuf_iterator<char>(input)),
                  std::istreambuf_iterator<char>());
    return source;
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr
        << "You have to pass 1 argument to the program, but none was passed."
        << std::endl;
    return -1;
  }

  const std::string filename(argv[1]);

  try {
    // Executa com o destrutor seguro para desalocar todos os ponteiros criados.
    ExecuteOpenCL exec(filename);
    exec.execute();
  } catch (const std::runtime_error &e) {
    throw e;
  }

  return 0;
}
