#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OpenCLUtils.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#define KERNELNAME "euclidean"


void computeDistanceTransformImage(const UCImage *image, float *imageOutput) {
  static const float thresholdVal = 1.0f;

  for(unsigned int x=0; x < image->attrs.width; ++x) {
    for(unsigned int y=0; y < image->attrs.height; ++y) {

      float minDistance = std::numeric_limits<float>::max();

      for(unsigned int ox=0; ox < image->attrs.width; ++ox) {
        for(unsigned int oy=0; oy < image->attrs.height; ++oy) {

          if(static_cast<float>(image->image[image->attrs.height*oy + ox]) >= thresholdVal) {
            const float distance = std::sqrt( 
              static_cast<float>((ox-x)*(ox-x)) + 
              static_cast<float>((oy-y)*(oy-y)) );

            if( distance < minDistance )
              minDistance = distance;
            
          }
        }
      }
      imageOutput[image->attrs.height*y + x] = minDistance;
    }
  }

}

unsigned char floatToPixVal(const float imageValue) {
  int tmpval = static_cast<int>(::std::floor(256 * imageValue));
  if (tmpval < 0) {
      return 0u;
  } else if (tmpval > 255) {
      return 255u;
  } else {
      return tmpval & 0xffu;
  }
}

class ExecuteDT {
public:
  ExecuteDT(const std::string &filename)
      : m_filename(filename), m_image(nullptr), m_output(nullptr){};

  void execute() {
    // As imagens esperadas são sempre com apenas um canal.
    int imageWidth, imageHeight;
    m_image = stbi_load(m_filename.c_str(), &imageWidth, &imageHeight,
                        nullptr, 1);
    if (m_image == nullptr)
      throw std::runtime_error("The image could not be loaded, please check if "
                               "the filename is corrected");

    const UCImage image = construcUCImage(m_image, imageHeight, imageWidth);
    const int imageSize = imageWidth * imageHeight;
    m_output = new float[imageSize];
    assert(m_output != nullptr);

    // Paralelo
    OpenCLUtils::executeOpenCL(KERNELNAME, ExecuteDT::readKernel(), &image,
                               m_output);

    // Sequencial
    //computeDistanceTransformImage(&image, m_output);
    // A imagem de saída para mostrar o resultado deve ser 
    unsigned char *imageOut = new unsigned char[imageSize];
    float maxValue = std::numeric_limits<float>::min();
    for (int i = 0; i < imageSize; i++) {
      if (maxValue < m_output[i])
        maxValue = m_output[i];
    }
    for (int i = 0; i < imageSize; i++) {
      imageOut[i] = floatToPixVal(m_output[i]/maxValue);
    }
    stbi_write_bmp("result.bmp", imageWidth, imageHeight, 1, imageOut);

    free(imageOut);
  }

  ~ExecuteDT() {
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
    ExecuteDT exec(filename);
    exec.execute();
  } catch (const std::runtime_error &e) {
    throw e;
  }

  return 0;
}
