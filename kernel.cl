/**
 * \brief Representação de coordenada 2d.
*/
typedef struct {
  unsigned int index;
  unsigned int y;
  unsigned int x;

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
float euclideanDistance(const Coordinate *coord1, const Coordinate *coord2) {
  return sqrt((float)((coord1->y - coord2->y)*(coord1->y - coord2->y) + (coord1->x - coord2->x)*(coord1->x - coord2->x)));
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
  unsigned char size;
  
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
  const Pixel pixel = neighborhood.pixels[index];

  return pixel;
}

typedef struct __attribute__ ((packed)) {
  unsigned int height;
  unsigned int width;
  unsigned int size;
} ImageAttrs;

/*
//Não pode ser passado para o kernel, uma vez que o OpenCL tem restrição quanto ao uso de
//arrays sem tamanho previamente especificado. Sessão 6.9 https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
typedef struct {
  ImageAttrs attrs; 
  unsigned char *image;
} UCImage;

UCImage construcUCImage(unsigned char *image, const unsigned int height, const unsigned int width) {
  UCImage ucimage;
  ucimage.image = image;
  ucimage.attrs.height = height;
  ucimage.attrs.width = width;
  ucimage.attrs.size = width * height;

  return ucimage;
}
*/

unsigned char getValueByCoord(const unsigned char *image, const ImageAttrs attrs, const Coordinate coord) {
  return image[coord.y * attrs.width + coord.x];
}

bool isBackgroudByCoord(const unsigned char *image, const ImageAttrs attrs, const Coordinate coord) {
  return getValueByCoord(image, attrs, coord) == 0;
}

Pixel getPixel(const unsigned char *image, const ImageAttrs attrs, int y, int x) {
  const Coordinate coord = constructCoord(y, x, attrs.width);
  return constructPixel(coord, isBackgroudByCoord(image, attrs, coord));
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const unsigned char *image, const ImageAttrs attrs, const Pixel *pixel) {
  Neighborhood neighborhood;
  for (int i = -1; i < 2; i++) {
    const unsigned int y = pixel->coord.y - i;
    if (y >= attrs.height)
      continue;

    for (int j = -1; i < 2; i++) {
      if (j == 0 && i == 0)
        continue;
      const unsigned int x = pixel->coord.x - j;
      if (x < 0 || x >= attrs.width)
        continue;


      addNeighbor(neighborhood, getPixel(image, attrs, y, x));
    }
  }
  
  return neighborhood;
}

typedef struct {
  Coordinate point;
  Coordinate nearestBackground;
} VoronoiDiagramMapEntry;

/*
//Não pode ser passado para o kernel, uma vez que o OpenCL tem restrição quanto ao uso de
//arrays sem tamanho previamente especificado. Sessão 6.9 https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
typedef struct {
  VoronoiDiagramMapEntry *entries;
  unsigned int sizeOfDiagram;
} VoronoiDiagramMap;
*/


int get_hash(const VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const Coordinate *key) {
  int result;
  result = key->index;
  return (result % diagramSize);
}

VoronoiDiagramMapEntry getVoronoiEntry(const VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const Coordinate *coord) {
  return map[get_hash(map, diagramSize, coord)];
}

void __kernel euclidean(
  __global const unsigned char *image,
  __global const ImageAttrs *imageAttrs,
  __global Pixel *pixelQueue,
  const unsigned int pixelQueueSize,
  __local Pixel *lpixelQueue,
  __global VoronoiDiagramMapEntry *voronoi
) {
  // Wavefront propagation
  __local unsigned int localPixelQueueOffset;
  localPixelQueueOffset = get_group_id(0)*(pixelQueueSize/get_num_groups(0));
  __local unsigned int localPixelQueueSize;
  localPixelQueueSize = get_group_id(0) != (get_num_groups(0) - 1) ? 
    (pixelQueueSize/get_num_groups(0)) 
    : 
    (pixelQueueSize/get_num_groups(0)) + (pixelQueueSize%get_num_groups(0));
  
  async_work_group_copy(lpixelQueue, (Pixel *)(pixelQueue+localPixelQueueOffset*localPixelQueueSize), localPixelQueueSize, 0);

  __private unsigned int privatePixelQueueOffset = get_local_id(0)*(localPixelQueueSize/get_local_size(0));
  __private unsigned int privatelPixelQueueSize = get_local_id(0) != (get_local_size(0) - 1) ? 
    (localPixelQueueSize/get_local_size(0)) 
    : 
    (localPixelQueueSize/get_local_size(0)) + (localPixelQueueSize%get_local_size(0));
  

}