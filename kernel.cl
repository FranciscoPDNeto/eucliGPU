/**
 * \brief A Coordenada é composta pelo valores de abscissa e ordenada, seguido do indice
 * geral.
*/
uint4 constructCoord(unsigned int y, unsigned int x, unsigned int imageWidth) {
  uint4 coord;
  coord.y = y;
  coord.x = x;
  coord.z = y*imageWidth + x;
  return coord;
}

uint4 constructInvalidCoord() {
  const unsigned int maxUInt = INFINITY;
  uint4 coord;
  coord.y = maxUInt;
  coord.x = maxUInt;
  return coord;
}

/**
 * \brief Calcula a distância euclideana.
*/
float euclideanDistance(const uint4 coord1, const uint4 coord2) {
  return sqrt((float)(coord1.y - coord2.y)*(coord1.y - coord2.y) + (coord1.x - coord2.x)*(coord1.x - coord2.x));
}

/**
 * \brief Constroi um pixel, representa uma coordenada e um valor.
*/
uint4 constructPixel(const uint4 coord, const uint value) {
  uint4 pixel;
  pixel.y = coord.y;
  pixel.x = coord.x;
  pixel.z = coord.z;
  pixel.w = value;
  
  return pixel;
}

typedef struct {
  // Há no máximo 8 vizinhos.
  uint4 pixels[8];
  // É unsigned char pois pode representar no máximo 8, então não é necessário
  // mais de um byte.
  ushort size;
  
} Neighborhood;

Neighborhood initNeighborhood() {
  Neighborhood neighborhood;
  neighborhood.size = 0;

  return neighborhood;
}

void addNeighbor(Neighborhood *neighborhood, uint4 pixel) {
  neighborhood->pixels[neighborhood->size++] = pixel;
}

uint4 getNeighbor(const Neighborhood neighborhood, const unsigned char index) {
  const uint4 pixel = neighborhood.pixels[index];

  return pixel;
}

typedef struct __attribute__ ((packed)) {
  uint height;
  uint width;
  uint size;
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

uchar getValueByCoord(const unsigned char *image, const ImageAttrs attrs, const uint4 coord) {
  return image[coord.y * attrs.width + coord.x];
}

bool isBackgroudByCoord(const unsigned char *image, const ImageAttrs attrs, const uint4 coord) {
  return getValueByCoord(image, attrs, coord) == 0;
}

uint4 getPixel(const unsigned char *image, const ImageAttrs attrs, const uint4 coordinate) {
  return constructPixel(coordinate, isBackgroudByCoord(image, attrs, coordinate));
}

uint4 getPixelByCoord(const unsigned char *image, const ImageAttrs attrs, int y, int x) {
  const uint4 coord = constructCoord(y, x, attrs.width);
  return getPixel(image, attrs, coord);
}

bool isBackgroudByPixel(const uint4 pixel) {
  return pixel.w;
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const unsigned char *image, __global const ImageAttrs attrs, const uint4 pixel) {
  Neighborhood neighborhood;
  neighborhood.size = 0;
  for (int i = -1; i < 2; i++) {
    const uint y = pixel.y - i;
    if (y >= attrs.height)
      continue;

    for (int j = -1; j < 2; j++) {
      if (j == 0 && i == 0)
        continue;
      const uint x = pixel.x - j;
      if (x < 0 || x >= attrs.width)
        continue;


      addNeighbor(&neighborhood, getPixelByCoord(image, attrs, y, x));
    }
  }
  
  return neighborhood;
}

typedef struct {
  uint4 point;
  uint4 nearestBackground;
} VoronoiDiagramMapEntry;

/*
//Não pode ser passado para o kernel, uma vez que o OpenCL tem restrição quanto ao uso de
//arrays sem tamanho previamente especificado. Sessão 6.9 https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
typedef struct {
  VoronoiDiagramMapEntry *entries;
  unsigned int sizeOfDiagram;
} VoronoiDiagramMap;
*/

int get_hash(__global const VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 key) {
  // Pega o valor do indice da coordenada.
  return (key.z %diagramSize);
}

__global VoronoiDiagramMapEntry *getVoronoiEntry(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 coord) {
  return &map[get_hash(map, diagramSize, coord)];
}

uint4 getVoronoiValue(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 coord) {
  return getVoronoiEntry(map, diagramSize, coord)->nearestBackground;
}

__global uint4 *getVoronoiValuePtr(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 coord) {
  return &(getVoronoiEntry(map, diagramSize, coord)->nearestBackground);
}

bool compareCoords(const uint4 coord1, const uint4 coord2) {
  return coord1.y == coord2.y && coord1.x == coord2.x && coord1.z == coord2.z;
}

void __kernel euclidean(
  __global const unsigned char *image,
  __global const ImageAttrs *imageAttrs,
  __global uint4 *pixelQueue,
  const unsigned int pixelQueueSize,
  __local uint4 *lpixelQueue,
  __global VoronoiDiagramMapEntry *voronoi,
  const unsigned int voronoiSize
) {
  // Wavefront propagation
  __local unsigned int localPixelQueueOffset;
  localPixelQueueOffset = get_group_id(0)*(pixelQueueSize/get_num_groups(0));
  __local unsigned int localPixelQueueSize;
  localPixelQueueSize = get_group_id(0) != (get_num_groups(0) - 1) ? 
    (pixelQueueSize/get_num_groups(0)) 
    : 
    (pixelQueueSize/get_num_groups(0)) + (pixelQueueSize%get_num_groups(0));
  
  async_work_group_copy(lpixelQueue, (pixelQueue+localPixelQueueOffset*localPixelQueueSize), localPixelQueueSize, 0);

  __private unsigned int privatePixelQueueOffset = get_local_id(0)*(localPixelQueueSize/get_local_size(0));
  __private unsigned int privatelPixelQueueSize = get_local_id(0) != (get_local_size(0) - 1) ? 
    (localPixelQueueSize/get_local_size(0)) 
    : 
    (localPixelQueueSize/get_local_size(0)) + (localPixelQueueSize%get_local_size(0));

  uint exceededPixelSize = 0;
  // o máximo de pixel excedido é a quantidade de pixels multiplicado pelo valor máximo de vizinhos
  // q cada píxel pode ter.
  uint4 exceededPixel[privatelPixelQueueSize*8];

  for (int i = 0; i < privatelPixelQueueSize; i++) {
    uint4 p = pixelQueue[i];
    uint4 coordinateP = constructCoord(p.y, p.x, p.z);

    Neighborhood neighborhood = getNeighborhood(image, imageAttrs, p);
    for (int j = 0; j < neighborhood.size; j++) {
      uint4 q = neighborhood.pixels[j];
      uint4 coordinateQ = constructCoord(q.y, q.x, q.z);
      do {
        uint4 curVRQ = getVoronoiValue(voronoi, voronoiSize, coordinateQ);

        if (euclideanDistance(q, getVoronoiValue(voronoi, voronoiSize, coordinateP)) < euclideanDistance(q, curVRQ)) {
          uint4 old = atomic_cmpxchg(getVoronoiValuePtr(voronoi, voronoiSize, coordinateQ), 
            curVRQ, getVoronoiValue(voronoi, voronoiSize, coordinateP));

          if (compareCoords(old, curVRQ)) {
            exceededPixel[exceededPixelSize++] = q;
            break;
          }
        } else
          break;
      } while (true);
    }
  }

  uint exceededPixelSizeQueue = exceededPixelSize;
  while(exceededPixelSizeQueue > 0) {
    uint exceededPixelSize = exceededPixelSizeQueue;
    for (int i = 0; i < exceededPixelSize; i++) {
      uint4 p = pixelQueue[i];
      exceededPixelSizeQueue--;
      uint4 coordinateP = constructCoord(p.y, p.x, p.z);

      Neighborhood neighborhood = getNeighborhood(image, imageAttrs, p);
      for (int j = 0; j < neighborhood.size; j++) {
        uint4 q = neighborhood.pixels[j];
        uint4 coordinateQ = constructCoord(q.y, q.x, q.z);
        do {
          uint4 curVRQ = getVoronoiValue(voronoi, voronoiSize, coordinateQ);

          if (euclideanDistance(q, getVoronoiValue(voronoi, voronoiSize, coordinateP)) < euclideanDistance(q, curVRQ)) {
            uint4 old = atomic_cmpxchg((volatile __global uint *)getVoronoiValuePtr(voronoi, voronoiSize, coordinateQ), 
              curVRQ, getVoronoiValue(voronoi, voronoiSize, coordinateP));

            if (compareCoords(old, curVRQ)) {
              exceededPixel[exceededPixelSizeQueue++] = q;
              break;
            }
          } else
            break;
        } while (true);
      }
    }
  }
  

}