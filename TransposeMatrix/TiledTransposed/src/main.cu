#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>


#define TILE_DIM 64
#define BLOCK_ROWS 8

// Otimizado com uint4: lê/escreve 4 bytes por vez
__global__ void transposeInPlaceTiledOpt(unsigned char *data, int width, int height, size_t pitch)
{
  __shared__ uint4 tileA[TILE_DIM][TILE_DIM/4 + 1];
  __shared__ uint4 tileB[TILE_DIM][TILE_DIM/4 + 1];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Diagonal tile: transpose in-place
  if (bx == by) {
    // Load 4 pixels at a time
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + ty + i;
      int col = bx * TILE_DIM + tx * 4;
      if (row < height && col < width - 3) {
        uint4 *rowPtr = (uint4 *)((char*)data + row * pitch);
        tileA[ty + i][tx] = rowPtr[col / 4];
      }
    }

    __syncthreads();

    // Write transposed
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + tx * 4;
      int col = bx * TILE_DIM + ty + i;
      if (row < height - 3 && col < width) {
        uint4 *rowPtr = (uint4 *)((char*)data + row * pitch);
        rowPtr[(col) / 4] = tileA[tx][ty + i];
      }
    }
  }
  // Off-diagonal tiles
  else if (bx < by) {
    // Load A and B
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + ty + i;
      int colA = bx * TILE_DIM + tx * 4;
      if (rowA < height && colA < width - 3) {
        uint4 *rowPtrA = (uint4 *)((char*)data + rowA * pitch);
        tileA[ty + i][tx] = rowPtrA[colA / 4];
      }

      int rowB = bx * TILE_DIM + ty + i;
      int colB = by * TILE_DIM + tx * 4;
      if (rowB < height && colB < width - 3) {
        uint4 *rowPtrB = (uint4 *)((char*)data + rowB * pitch);
        tileB[ty + i][tx] = rowPtrB[colB / 4];
      }
    }

    __syncthreads();

    // Write swap
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + tx * 4;
      int colA = bx * TILE_DIM + ty + i;
      if (rowA < height - 3 && colA < width) {
        uint4 *rowPtrA = (uint4 *)((char*)data + rowA * pitch);
        rowPtrA[colA / 4] = tileB[tx][ty + i];
      }

      int rowB = bx * TILE_DIM + tx * 4;
      int colB = by * TILE_DIM + ty + i;
      if (rowB < height - 3 && colB < width) {
        uint4 *rowPtrB = (uint4 *)((char*)data + rowB * pitch);
        rowPtrB[colB / 4] = tileA[tx][ty + i];
      }
    }
  }
}


#define DEVICE_MALLOC_LIMIT 1024*1024*1024


__global__ void transposeInPlaceTiled(unsigned char *data, int width, int height, size_t pitch)
{
  __shared__ unsigned char tileA[TILE_DIM][TILE_DIM + 1];
  __shared__ unsigned char tileB[TILE_DIM][TILE_DIM + 1];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Only handle square matrices (caller ensures width==height)

  // Diagonal tile: transpose in-place
  if (bx == by) {
    // Load tile into shared memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + ty + i;
      int col = bx * TILE_DIM + tx;
      if (row < height && col < width) {
        unsigned char *rowPtr = (unsigned char *)((char*)data + row * pitch);
        tileA[ty + i][tx] = rowPtr[col];
      }
    }

    __syncthreads();

    // Write transposed back
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + tx;
      int col = bx * TILE_DIM + ty + i;
      if (row < height && col < width) {
        unsigned char *rowPtr = (unsigned char *)((char*)data + row * pitch);
        rowPtr[col] = tileA[tx][ty + i];
      }
    }
  }
  // Off-diagonal tiles: swap tile (bx,by) with transposed tile (by,bx)
  else if (bx < by) {
    // Load A = tile(bx,by) and B = tile(by,bx)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + ty + i;
      int colA = bx * TILE_DIM + tx;
      if (rowA < height && colA < width) {
        unsigned char *rowPtrA = (unsigned char *)((char*)data + rowA * pitch);
        tileA[ty + i][tx] = rowPtrA[colA];
      } else {
        tileA[ty + i][tx] = 0;
      }

      int rowB = bx * TILE_DIM + ty + i;
      int colB = by * TILE_DIM + tx;
      if (rowB < height && colB < width) {
        unsigned char *rowPtrB = (unsigned char *)((char*)data + rowB * pitch);
        tileB[ty + i][tx] = rowPtrB[colB];
      } else {
        tileB[ty + i][tx] = 0;
      }
    }

    __syncthreads();

    // Write transposed: A <- transpose(B), B <- transpose(A)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + ty + i;
      int colA = bx * TILE_DIM + tx;
      // write into A region from tileB transposed
      if (rowA < height && colA < width) {
        unsigned char *rowPtrA = (unsigned char *)((char*)data + rowA * pitch);
        rowPtrA[colA] = tileB[tx][ty + i];
      }

      int rowB = bx * TILE_DIM + ty + i;
      int colB = by * TILE_DIM + tx;
      // write into B region from tileA transposed
      if (rowB < height && colB < width) {
        unsigned char *rowPtrB = (unsigned char *)((char*)data + rowB * pitch);
        rowPtrB[colB] = tileA[tx][ty + i];
      }
    }
  }
}




// filename deve ser de um arquivo pgm
// Retorna um ponteiro para GPU
unsigned char* loadPGMToDevice(char* filename, int& width, int &height, size_t &pitch) {
  if(strcmp(&(filename[strlen(filename)-3]), "pgm")) {
    printf("Imagem inválida\n");
    exit(1);
  }

  // Variáveis auxiliares
  int i;
  unsigned char* d_imagem_row;

  // Abrimos a imagem
  FILE* imagem;
  imagem = fopen(filename, "rb");
  if(imagem==NULL) {
    printf("Falha ao ler a imagem\n");
    exit(1);
  }

  // Buffer de leitura do header
  char header_buffer[1024];

  // Variáveis do header
  char file_type[50];
  unsigned short max_val;

  // Lendo o header
  memset(header_buffer, 0, 1024);
  fgets(header_buffer, 1024, imagem);
  sscanf(header_buffer, "%s\n", file_type);

  // Lemos as dimensões da imagem
  memset(header_buffer, 0, strlen(header_buffer));
  fgets(header_buffer, 1024, imagem);
  if(sscanf(header_buffer, "%d %d\n", &height, &width) < 2) {
    printf("Falha ao ler dimsnesões da imagem\n");
    exit(1);
  }

  // Lemos o maior valor na imagem 
  memset(header_buffer, 0, strlen(header_buffer));
  fgets(header_buffer, 1024, imagem);
  sscanf(header_buffer, "%hu\n", &max_val); 

  if(max_val != 255) {
    printf("Pixels de 16-bits não são suportados\n");
    exit(1);
  }

  unsigned char* d_imagem;

  cudaError_t err = cudaMallocPitch(&d_imagem, &pitch, width*sizeof(uchar), height);
  if(err) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  uchar* buffer = (unsigned char*) malloc(width*sizeof(uchar)+32);

  for(i=0; i<height; i++) {
    if(fread(buffer, sizeof(unsigned char), width, imagem) < width) {
      printf("Falha na leitura (iteracao=%d)\n", i);
      exit(1);
    }
    
    d_imagem_row = (uchar*)((char*)d_imagem + i*pitch);
    err = cudaMemcpy(d_imagem_row, buffer, width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    if(err) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(1);
    }

  }
  

  fclose(imagem);
  free(buffer);

  return d_imagem;
}

void configureHeapSize() {
    // Aumentamos o heap para acomodar a fila
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_LIMIT);
}

int main(int argc, char** argv) {
    
    if(argc < 5) {
      printf("Numero insuficiente de argumentos (%d/4)\n", argc-1);
      return 1;
    }

    char* marker_filename = argv[1];
    char* mask_filename = argv[2];
    const char* output_filename = argv[3];
    int usePGMLoader = atoi(argv[4]);

    // Informações das imagens
    int size;
    int height;

    // Opencv container de imagem
    cv::Mat marker_img, mask_img;
    
    // Containers das imagens
    unsigned char* d_marker;
    size_t pitchMarker;
    unsigned char* d_mask;
    size_t pitchMask;
    
    // Configurando o heap do device
    configureHeapSize();
    
    // Lendo as imagens
	  // Usamos o carregamento customizado de pgm
    if(usePGMLoader) {
      d_marker = loadPGMToDevice(marker_filename, size, height, pitchMarker);
    }
    // Usamos o opencv
    else {
      marker_img = cv::imread(marker_filename, cv::IMREAD_GRAYSCALE);
      
      if(marker_img.empty()) {
          printf("Falha ao ler o marcador\n");
          exit(1);
      }
      
      size = marker_img.size().width;
      // Alocamos a imagem
      cudaMallocPitch(&d_marker, &pitchMarker, size*sizeof(unsigned char), size);

      // Enviamos a imagem para a GPU
      cudaMemcpy2D(d_marker, pitchMarker, marker_img.data, marker_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);

    }

	  // Usamos o carregamento customizado de pgm
    if(usePGMLoader) {
      d_mask = loadPGMToDevice(mask_filename, size, height, pitchMask);
    }
    // Usamos o opencv
    else {
	    mask_img = cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);

      if(mask_img.empty()) {
          printf("Falha ao ler a máscara\n");
          exit(1);
      }

      // Alocamos as imagens
      cudaMallocPitch(&d_mask, &pitchMask, size*sizeof(unsigned char), size);

      // Enviamos a imagem para a GPU
      cudaMemcpy2D(d_mask, pitchMask, mask_img.data, mask_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);
    }
    printf("size: %d\n", size);
    cudaDeviceSynchronize();
    
    auto tStart = std::chrono::high_resolution_clock::now();

    // Configure launch dimensions and run in-place transpose with coalesced reads
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((size + TILE_DIM - 1) / TILE_DIM, (size + TILE_DIM - 1) / TILE_DIM);
    transposeInPlaceTiledOpt<<<dimGrid, dimBlock>>>(d_marker, size, size, pitchMarker);
    cudaError_t kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(kernErr));
      return 1;
    }
    cudaDeviceSynchronize();

    auto tEnd = std::chrono::high_resolution_clock::now();

    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("Tempo GPU: %ld ms\n", ms_int.count());

    //printf("%ld\n", ms_int.count());

    cv::Mat output_img(cv::Size(size, size), CV_8UC1);
    // Enviamos o resultado para o host
    // cudaMemcpy2D(output_img.data, output_img.step, 
    //                 d_marker, pitchMarker, 
    //                 size * sizeof(unsigned char), size, 
    //                 cudaMemcpyDeviceToHost);

    // cv::imwrite(output_filename, output_img);

    cudaFree(d_marker);
    //cudaFree(d_mask);
    
    return 0;
}
