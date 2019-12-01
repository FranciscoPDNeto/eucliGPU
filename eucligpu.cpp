#include <fstream>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OpenCLUtils.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#define KERNELNAME "euclidean"


void sequentialDT(const UCImage *image, float *imageOutput) {

  for(unsigned int x=0; x < image->attrs.v2[1]; ++x) {
    for(unsigned int y=0; y < image->attrs.v2[0]; ++y) {

      float minDistance = std::numeric_limits<float>::max();

      cl_uint4 coord1 = constructCoord(y, x, image->attrs.v2[1]);
      for(unsigned int innerX = 0; innerX < image->attrs.v2[1]; ++innerX) {
        for(unsigned int innerY = 0; innerY < image->attrs.v2[0]; ++innerY) {

          const cl_uint4 coord2 = constructCoord(innerY, innerX, image->attrs.v2[1]);

          if(!isBackgroudByCoord(image, coord2)) {
            const float distance = euclideanDistance(coord1, coord2);

            if(distance < minDistance)
              minDistance = distance;
            
          }
        }
      }
      imageOutput[image->attrs.v2[1]*y + x] = minDistance;
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

template<typename T>
std::vector<std::vector<T>> SplitVector(const std::vector<T>& vec, size_t n) {
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
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
    m_output = new float[imageSize]();
    assert(m_output != nullptr);

    // Paralelo
    // Initialization
    
    VoronoiDiagramMap voronoi;
    voronoi.sizeOfDiagram = imageSize;
    voronoi.entries = new VoronoiDiagramMapEntry[voronoi.sizeOfDiagram];
    // É usado o vector pois é mais fácil extrair o array primitivo para se passar a
    // posteriori ao kernel.
    std::vector<cl_uint4> queue;
    for (int x = 0; x < imageWidth; x++)
      for (int y = 0; y < imageHeight; y++) {
        cl_uint4 coordinate = constructCoord(y, x, imageWidth);
        Neighborhood neighborhood = getNeighborhood(&image, getPixel(&image, coordinate));

        if (isBackgroudByCoord(&image, coordinate)) {
          voronoi.entries[coordinate.v4[2]] =
            VoronoiDiagramMapEntry{ coordinate, coordinate };

          for (int i = 0; i < neighborhood.size; i++) {
            const cl_uint4 pixel = neighborhood.pixels[i];
            if (!isBackgroudByCoord(&image, pixel)) {
              queue.push_back(coordinate);
            }
          }
        } else {
          voronoi.entries[coordinate.v4[2]] = 
            VoronoiDiagramMapEntry{ coordinate, constructInvalidCoord() };
        }
      }
    if (!queue.empty()) {
      // Wavefront propagation
      OpenCLUtils::executeOpenCL(KERNELNAME, ExecuteDT::readKernel(), &image,
                                queue, &voronoi, m_output);
    }

    // Distance calculation
    float maxDistance = std::sqrt(std::pow(imageWidth, 2) + std::pow(imageHeight, 2));
    unsigned char *imageOut = new unsigned char[imageSize];
    for (int x = 0; x < imageWidth; x++)
      for (int y = 0; y < imageHeight; y++) {
        cl_uint4 coordinate = constructCoord(y, x, imageWidth);
        cl_uint4 nearest = voronoi
          .entries[coordinate.v4[2]]
          .nearestBackground;
        float distance = euclideanDistance(coordinate, nearest);
        imageOut[coordinate.v4[2]] = floatToPixVal(distance / maxDistance);
      }
    free(voronoi.entries);
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
