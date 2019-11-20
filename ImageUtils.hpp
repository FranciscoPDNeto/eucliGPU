#include <cassert>
#include <math.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

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

Coordinate constructInvalidCoord() {
  Coordinate coord;
  coord.index = -1;
  coord.y = -1;
  coord.x = -1;

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
  cl_ushort size;
  
} Neighborhood;

Neighborhood initNeighborhood() {
  Neighborhood neighborhood;
  neighborhood.size = 0;

  return neighborhood;
}

void addNeighbor(Neighborhood *neighborhood, Pixel pixel) {
  neighborhood->pixels[neighborhood->size++] = pixel;
}

Pixel getNeighbor(const Neighborhood neighborhood, const unsigned char index) {
  assert(index < neighborhood.size);
  const Pixel pixel = neighborhood.pixels[index];

  return pixel;
}

typedef struct __attribute__ ((packed)) {
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

Pixel getPixel(const UCImage *image, const Coordinate coordinate) {
  return constructPixel(coordinate, isBackgroudByCoord(image, coordinate));
}

Pixel getPixelByCoord(const UCImage *image, cl_int y, cl_int x) {
  const Coordinate coord = constructCoord(y, x, image->attrs.width);
  return getPixel(image, coord);
}

bool isBackgroudByPixel(const Pixel pixel) {
  return pixel.background;
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const UCImage *image, const Pixel& pixel) {
  Neighborhood neighborhood;
  neighborhood.size = 0;
  for (cl_int i = -1; i < 2; i++) {
    const cl_uint y = pixel.coord.y - i;
    if (y >= image->attrs.height)
      continue;

    for (cl_int j = -1; j < 2; j++) {
      if (j == 0 && i == 0)
        continue;
      const cl_uint x = pixel.coord.x - j;
      if (x < 0 || x >= image->attrs.width)
        continue;


      addNeighbor(&neighborhood, getPixelByCoord(image, y, x));
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