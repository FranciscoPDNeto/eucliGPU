
typedef struct __attribute__ ((packed)) {
  unsigned int height;
  unsigned int width;
  unsigned int size;
} ImageAttrs;

__kernel void DistanceTransform(
  __global const float* vIn, 
  __global const ImageAttrs *imageAttrs,
  __global float*       vOut)
{
  int iGID = get_global_id(0);

  int iDx = imageAttrs->height;
  int iDy = imageAttrs->width;

  if (iGID >= (iDx*iDy))
  {   
    return; 
  }

  float minVal = MAXFLOAT;
  int minX = 0;
  int minY = 0;
  for(int y = 0; y < iDy; y++)  
  {
    for(int x = 0; x < iDx; x++)
    {      
      if(vIn[y*iDy + x] >= 1.0f)
      {
        int idX = iGID % iDy;
        int idY = iGID / iDy;
        float dist = sqrt( (float)((idX-x)*(idX-x) + (idY-y)*(idY-y)) );

        if(dist < minVal)
        {
          minVal = dist;
        }
      }
    }
  }

  vOut[iGID] = minVal;
}