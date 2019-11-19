#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OpenCLUtils.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#define KERNELNAME "euclidean"

/**
 * \brief Representação de coordenada 2d.
*/
typedef struct {
  cl_uint index;
  cl_uint y;
  cl_uint x;

} Coordinate;

Coordinate constructCoord(unsigned int y, unsigned int x, unsigned int imageWidth) {
  Coordinate coord;
  coord.index = y*imageWidth + x;
  coord.y = y;
  coord.x = x;

  return coord;
}

/**
 * \brief Calcula a distância euclideana.
*/
cl_float euclideanDistance(const Coordinate& coord1, const Coordinate& coord2) {
  return std::sqrt(std::pow(coord1.y - coord2.y, 2) + std::pow(coord1.x - coord2.x, 2));
}

/**
 * \brief Representa o Pixel da imagem.
*/
typedef struct {
  Coordinate coord;
  // Valor que corresponde a true caso seja célula de fundo, ou false(0) caso seja célula de frente(1).
  bool background;
} Pixel;

Pixel constructPixel(const Coordinate coord, const bool background) {
  Pixel pixel;
  pixel.coord = coord;
  pixel.background = background;
  
  return pixel;
}

typedef struct {
  // Há no máximo 8 vizinhos.
  Pixel pixels[8];
  // É unsigned char pois pode representar no máximo 8, então não é necessário
  // mais de um byte.
  cl_uchar size;
  
} Neighborhood;

Neighborhood initNeighborhood() {
  Neighborhood neighborhood;
  neighborhood.size = 0;

  return neighborhood;
}

void addNeighbor(Neighborhood neighborhood, Pixel pixel) {
  neighborhood.pixels[neighborhood.size++] = pixel;
}

Pixel getNeighbor(const Neighborhood neighborhood, const unsigned char index) {
  assert(index < neighborhood.size);
  const Pixel pixel = neighborhood.pixels[index];

  return pixel;
}

typedef struct {
  cl_uint height;
  cl_uint width;
  cl_uint size;
} ImageAttrs;

typedef struct {
  ImageAttrs attrs; 
  cl_uchar *image;
} UCImage;


UCImage construcUCImage(unsigned char *image, const unsigned int height, const unsigned int width) {
  UCImage ucimage;
  ucimage.image = image;
  ucimage.attrs.height = height;
  ucimage.attrs.width = width;
  ucimage.attrs.size = width * height;

  return ucimage;
}

cl_uchar getValueByCoord(const UCImage *image, const Coordinate coord) {
  return image->image[coord.y * image->attrs.width + coord.x];
}

bool isBackgroudByCoord(const UCImage *image, const Coordinate coord) {
  return getValueByCoord(image, coord) == 0;
}

Pixel getPixel(const UCImage *image, cl_int y, cl_int x) {
  const Coordinate coord = constructCoord(y, x, image->attrs.width);
  return constructPixel(coord, isBackgroudByCoord(image, coord));
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const UCImage *image, const Pixel& pixel) {
  Neighborhood neighborhood;
  for (cl_int i = -1; i < 2; i++) {
    const cl_uint y = pixel.coord.y - i;
    if (y >= image->attrs.height)
      continue;

    for (cl_int j = -1; i < 2; i++) {
      if (j == 0 && i == 0)
        continue;
      const cl_uint x = pixel.coord.x - j;
      if (x < 0 || x >= image->attrs.width)
        continue;


      addNeighbor(neighborhood, getPixel(image, y, x));
    }
  }
  
  return neighborhood;
}

typedef struct {
  Coordinate point;
  Coordinate nearestBackground;
} VoronoiDiagramMapEntry;

typedef struct {
  VoronoiDiagramMapEntry *entries;
  cl_uint sizeOfDiagram;
} VoronoiDiagramMap;


int get_hash(const VoronoiDiagramMap *map, const Coordinate *key) {
  int result;
  result = key->index;
  return (result % map->sizeOfDiagram);
}

VoronoiDiagramMapEntry getVoronoiEntry(const VoronoiDiagramMap *map, const Coordinate *coord) {
  return map->entries[get_hash(map, coord)];
}

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
    //OpenCLUtils::executeOpenCL(KERNELNAME, ExecuteOpenCL::readKernel(), m_image,
                               //imageSize, m_output, imageSize);

    // Sequencial
    computeDistanceTransformImage(&image, m_output);
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
