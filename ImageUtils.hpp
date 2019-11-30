#include <cassert>
#include <math.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

/**
 * \brief A Coordenada é composta pelo valores de abscissa e ordenada, seguido do indice
 * geral.
*/
cl_uint4 constructCoord(unsigned int y, unsigned int x, unsigned int imageWidth) {
  cl_uint4 coord{y, x, y*imageWidth + x};

  return coord;
}

cl_uint4 constructInvalidCoord() {
  const unsigned int maxUInt = std::numeric_limits<unsigned int>::max();
  cl_uint4 coord{maxUInt, maxUInt};
  return coord;
}

/**
 * \brief Calcula a distância euclideana.
*/
cl_float euclideanDistance(const cl_uint4& coord1, const cl_uint4& coord2) {
  return std::sqrt(std::pow(coord1.v4[0] - coord2.v4[0], 2) + std::pow(coord1.v4[1] - coord2.v4[1], 2));
}

/**
 * \brief Constroi um pixel, representa uma coordenada e um valor.
*/
cl_uint4 constructPixel(const cl_uint4 coord, const cl_uint value) {
  cl_uint4 pixel{coord.v4[0], coord.v4[1], coord.v4[2], value};
  
  return pixel;
}

typedef struct {
  // Há no máximo 8 vizinhos.
  cl_uint4 pixels[8];
  // É unsigned char pois pode representar no máximo 8, então não é necessário
  // mais de um byte.
  cl_ushort size;
  
} Neighborhood;

Neighborhood initNeighborhood() {
  Neighborhood neighborhood;
  neighborhood.size = 0;

  return neighborhood;
}

void addNeighbor(Neighborhood *neighborhood, cl_uint4 pixel) {
  neighborhood->pixels[neighborhood->size++] = pixel;
}

cl_uint4 getNeighbor(const Neighborhood neighborhood, const unsigned char index) {
  assert(index < neighborhood.size);
  const cl_uint4 pixel = neighborhood.pixels[index];

  return pixel;
}

typedef struct {
  cl_uint2 attrs; 
  cl_uchar *image;
} UCImage;


UCImage construcUCImage(unsigned char *image, const unsigned int height, const unsigned int width) {
  UCImage ucimage;
  ucimage.image = image;
  ucimage.attrs.v2[0] = height;
  ucimage.attrs.v2[1] = width;

  return ucimage;
}

cl_uchar getValueByCoord(const UCImage *image, const cl_uint4 coord) {
  return image->image[coord.v4[0] * image->attrs.v2[1] + coord.v4[1]];
}

bool isBackgroudByCoord(const UCImage *image, const cl_uint4 coord) {
  return getValueByCoord(image, coord) == 0;
}

cl_uint4 getPixel(const UCImage *image, const cl_uint4 coordinate) {
  return constructPixel(coordinate, isBackgroudByCoord(image, coordinate));
}

cl_uint4 getPixelByCoord(const UCImage *image, cl_int y, cl_int x) {
  const cl_uint4 coord = constructCoord(y, x, image->attrs.v2[1]);
  return getPixel(image, coord);
}

bool isBackgroudByPixel(const cl_uint4 pixel) {
  return pixel.v4[3];
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const UCImage *image, const cl_uint4& pixel) {
  Neighborhood neighborhood;
  neighborhood.size = 0;
  for (cl_int i = -1; i < 2; i++) {
    const cl_uint y = pixel.v4[0] - i;
    if (y >= image->attrs.v2[0])
      continue;

    for (cl_int j = -1; j < 2; j++) {
      if (j == 0 && i == 0)
        continue;
      const cl_uint x = pixel.v4[1] - j;
      if (x < 0 || x >= image->attrs.v2[1])
        continue;


      addNeighbor(&neighborhood, getPixelByCoord(image, y, x));
    }
  }
  
  return neighborhood;
}

typedef struct {
  cl_uint4 point;
  cl_uint4 nearestBackground;
} VoronoiDiagramMapEntry;

typedef struct {
  VoronoiDiagramMapEntry *entries;
  cl_uint sizeOfDiagram;
} VoronoiDiagramMap;


int get_hash(const VoronoiDiagramMap *map, const cl_uint4 *key) {
  // Pega o valor do indice da coordenada.
  return (key->v4[2] % map->sizeOfDiagram);
}

VoronoiDiagramMapEntry getVoronoiEntry(const VoronoiDiagramMap *map, const cl_uint4 *coord) {
  return map->entries[get_hash(map, coord)];
}