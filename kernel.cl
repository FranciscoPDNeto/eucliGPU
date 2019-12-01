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

/*
//Não pode ser passado para o kernel, uma vez que o OpenCL tem restrição quanto ao uso de
//arrays sem tamanho previamente especificado. Sessão 6.9 https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
typedef struct {
  ImageAttrs attrs; 
  unsigned char *image;
} UCImage;

UCImage constructUCImage(unsigned char *image, const unsigned int height, const unsigned int width) {
  UCImage ucimage;
  ucimage.image = image;
  ucimage.attrs.height = height;
  ucimage.attrs.width = width;
  ucimage.attrs.size = width * height;

  return ucimage;
}
*/

uchar getValueByCoord(const __global unsigned char *image, const uint2 attrs, const uint4 coord) {
  return image[coord.y * attrs.x + coord.x];
}

bool isBackgroudByCoord(const __global unsigned char *image, const uint2 attrs, const uint4 coord) {
  return getValueByCoord(image, attrs, coord) == 0;
}

uint4 getPixel(const __global unsigned char *image, const uint2 attrs, const uint4 coordinate) {
  return constructPixel(coordinate, isBackgroudByCoord(image, attrs, coordinate));
}

uint4 getPixelByCoord(const __global unsigned char *image, const uint2 attrs, int y, int x) {
  const uint4 coord = constructCoord(y, x, attrs.x);
  return getPixel(image, attrs, coord);
}

bool isBackgroudByPixel(const uint4 pixel) {
  return pixel.w;
}

/**
 * \brief Retorna a lista de valores vizinhos ao pixel na imagem.
 * A vizinhança utilizada é a simples, vizinhança direta na janela 3x3.
*/
Neighborhood getNeighborhood(const __global unsigned char *image, const uint2 attrs, const uint4 pixel) {
  Neighborhood neighborhood;
  neighborhood.size = 0;
  for (int i = -1; i < 2; i++) {
    const uint y = pixel.y - i;
    if (y >= attrs.y)
      continue;

    for (int j = -1; j < 2; j++) {
      if (j == 0 && i == 0)
        continue;
      const uint x = pixel.x - j;
      if (x < 0 || x >= attrs.x)
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

__global VoronoiDiagramMapEntry *getVoronoiEntry(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 pixel) {
  return &map[get_hash(map, diagramSize, pixel)];
}

uint4 getVoronoiValue(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 pixel) {
  return getVoronoiEntry(map, diagramSize, pixel)->nearestBackground;
}

volatile __global uint4 *getVoronoiValuePtr(__global VoronoiDiagramMapEntry *map, const unsigned int diagramSize, const uint4 pixel) {
  return &(getVoronoiEntry(map, diagramSize, pixel)->nearestBackground);
}

bool compareCoords(const uint4 coord1, const uint4 coord2) {
  return coord1.y == coord2.y && coord1.x == coord2.x && coord1.z == coord2.z;
}

bool push(__private uint4 *stack, uint4 value) {
  // pushing from bot, so you can pop it from top later (FIFO)
  // circular buffer for top performance
  uint bufLen=100*100*8*10;
  // zeroth element is counter for newest added element
  // first element is oldest element


  // circular buffer 
  uint nextIndex=(stack[0].x%bufLen+1); // +1 because of top-bot headers

  // if overflows, it overwrites oldest elements one by one
  stack[nextIndex]=value;


  // if overflows, it still increments 
  stack[0].x++;
  //printf("stack x: %d\n", stack[0].x);

  // simple and fast
  return true;
}

bool empty(__private uint4 * stack) {
  // tricky if you overflow both
  return (stack[0].x==stack[0].y);
}

uint size(__private uint4 *stack) {
  return (stack[0].y - stack[0].x);
}

uint4 front(__private uint4 *stack) {
  uint bufLen=100*100*8*10;

  // oldest element value (top)
  uint ptr=stack[0].y%bufLen+1; // circular adr + 1 header


  return stack[ptr];
}

uint4 pop(__private uint4 *stack) {
  uint bufLen=100*100*8*10;
  uint ptr=stack[0].y%bufLen+1;
  // pop from top (oldest)
  uint4 returnValue=stack[ptr];
  stack[ptr]=0;

  // this will be new top ctr for ptr
  stack[0].y++;
  //printf("stack y: %d\nstack x: %d\n", stack[0].y, stack[0].x);

  // if underflows, gets garbage, don't underflow

  return returnValue;
}

void __kernel euclidean(
  __global const unsigned char *image,
  const uint2 imageAttrs,
  __global uint4 *pixelQueue,
  const unsigned int pixelQueueSize,
  __local uint4 *lpixelQueue,
  __global VoronoiDiagramMapEntry *voronoi,
  const unsigned int voronoiSize
) {
  //printf("PixelQueueSize: %d\n", pixelQueueSize);
  // Wavefront propagation
  __local unsigned int localPixelQueueOffset;
  localPixelQueueOffset = get_group_id(0)*(pixelQueueSize/get_num_groups(0));
  __local unsigned int localPixelQueueSize;
  localPixelQueueSize = get_group_id(0) != (get_num_groups(0) - 1) ? 
    (pixelQueueSize/get_num_groups(0)) 
    : 
    (pixelQueueSize/get_num_groups(0)) + (pixelQueueSize%get_num_groups(0));
  
  //printf("LocalPixelQueueSize: %d\n", localPixelQueueSize);
  //printf("Group: %d\n", get_group_id(0));
  
  async_work_group_copy(lpixelQueue, (pixelQueue+localPixelQueueOffset*get_group_id(0)), localPixelQueueSize, 0);

  __private unsigned int privatePixelQueueOffset = get_local_id(0)*(localPixelQueueSize/get_local_size(0));
  __private unsigned int privatePixelQueueSize = get_local_id(0) != (get_local_size(0) - 1) ? 
    (localPixelQueueSize/get_local_size(0)) 
    : 
    (localPixelQueueSize/get_local_size(0)) + (localPixelQueueSize%get_local_size(0));
  
  uint exceededPixelSize = 0;
  // o máximo de pixel excedido é a quantidade de pixels multiplicado pelo valor máximo de vizinhos
  // q cada píxel pode ter.
  // Fila circular de pixels excedidos com tamanho máximo de 100, e o primeiro elemento x e y indica o inicio e fim
  // da fila respectivamente.
  uint4 exceededPixel[100*100*8];
  exceededPixel[0].x = 0;
  exceededPixel[0].y = 0;

  for (int i = 0; i < privatePixelQueueSize; i++) {
    uint4 p = lpixelQueue[privatePixelQueueOffset + i];
    __global uint4 *area = &(voronoi[p.z].nearestBackground);
    //uint4 coordinateP = constructCoord(p.y, p.x, p.z);

    Neighborhood neighborhood = getNeighborhood(image, imageAttrs, p);
    for (int j = 0; j < neighborhood.size; j++) {
      uint4 q = neighborhood.pixels[j];
      //uint4 coordinateQ = constructCoord(q.y, q.x, q.z);
      __global uint4 *curVRQ = &(voronoi[q.z].nearestBackground);
      volatile __global uint4 *voronoiValuePtr = getVoronoiValuePtr(voronoi, voronoiSize, q);
      if (euclideanDistance(q, p) < euclideanDistance(q, *curVRQ)) {
        do {

            uint4 old;
            //printf("Voronoi: %d %d %d", voronoiValuePtr->x, voronoiValuePtr->y, voronoiValuePtr->z);
            *voronoiValuePtr = *area;
            /*
            old.x = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->x, 
              curVRQ.x, area->x);
            old.y = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->y, 
              curVRQ.y, area->y);
            old.z = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->z, 
              curVRQ.z, area->z);
            //printf("Voronoi after GPU: %d %d %d", voronoiValuePtr->x, voronoiValuePtr->y, voronoiValuePtr->z);

            if (compareCoords(old, curVRQ)) {
              
              //exceededPixel[exceededPixelSize++] = q;
              //push(exceededPixel, q);
              printf("tchau");
              break;
            }
            */
            push(exceededPixel, q);
        } while (false);
      }
    }
  }

  //int depth = 0;
  while(!empty(exceededPixel)) {
    uint4 p = pop(exceededPixel);
    __global uint4 *area = &(voronoi[p.z].nearestBackground);
    
    Neighborhood neighborhood = getNeighborhood(image, imageAttrs, p);
    for (int j = 0; j < neighborhood.size; j++) {
      uint4 q = neighborhood.pixels[j];
      //uint4 coordinateQ = constructCoord(q.y, q.x, q.z);
      do {
        __global uint4 *curVRQ = &(voronoi[q.z].nearestBackground);

        if (euclideanDistance(q, p) < euclideanDistance(q, *curVRQ)) {
          uint4 old;
          volatile __global uint4 *voronoiValuePtr = getVoronoiValuePtr(voronoi, voronoiSize, q);
          *voronoiValuePtr = *area;
          
          //printf("Voronoi: %d %d %d", voronoiValuePtr->x, voronoiValuePtr->y, voronoiValuePtr->z);
          /*
          old.x = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->x, 
            curVRQ.x, area->x);
          old.y = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->y, 
            curVRQ.y, area->y);
          old.z = atomic_cmpxchg((volatile __global uint*)voronoiValuePtr->z, 
            curVRQ.z, area->z);
          //printf("Voronoi after GPU: %d %d %d", voronoiValuePtr->x, voronoiValuePtr->y, voronoiValuePtr->z);

          if (compareCoords(old, curVRQ)) {
            //exceededPixel[exceededPixelSize++] = q;
            //push(exceededPixel, q);
            break;
          }
        } else
          break;
          */
          //if (depth < 90000) {
            push(exceededPixel, q);
            //printf("enqueing element %d %d for element %d %d\n", q.x, q.y, p.x, p.y);
            //depth++;
          //}
        }
      } while (false);
    }
  }
}