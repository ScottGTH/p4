#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel
__global__ void 
conv2d_kernel(const float *d_input, const float *d_filter, float *d_output, int f_dimen, int h, int w, int padded_width, int numElements, int pad_total, int pad_total_orig, int n, int k, int padded_height)
{
  int toto = padded_width*(h-1) + w;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < toto && ((i % padded_width) < w))
  {
    
    
    for(int z = 0; z < n; z++){
      for(int S = 0; S < k; S++){
        int a = 0;
        float sum = (float)a;
        int g = 0;
        int r = 0;
        for(int j = 0; j < f_dimen; j++){
          for(int x = 0; x < f_dimen; x++){
            g = i + x + padded_width*j + z*padded_width*padded_height;
            r = x + f_dimen*j + S*f_dimen*f_dimen;
            sum += d_filter[r] * d_input[g];
          }
        }
        //int I = i + z*padded_width*padded_height; 
        int Q = i / padded_width;
        int L = z*w*h*k + w*(Q) + i % padded_width + S*w*h;

        d_output[L] = sum;
      }
    }
  }
}


int main(int argc, char *argv[]) {

  // Read the inputs from command line

  // Allocate/move data using cudaMalloc and cudaMemCpy

  // Launch the kernel

  // Print the output

  // Clean up the memory

  
  cudaError_t err = cudaSuccess;
  
	char *trace_file1;
  char *trace_file2;  
  trace_file1 = argv[1];
  trace_file2 = argv[2];
  std::ifstream file_in(trace_file1);
  std::ifstream file_filter(trace_file2);
  std::string line;

  int h = 0;
  int w = 0;
  int n = 0;
	
	//std::ifstream        file_in, file_filter;          
	//std::vector<float> inp;
	int *hw= (int *)malloc(2*sizeof(int));
	int numElements=0;
	
	//file_in.open("input.txt", std::ifstream::in);
	std::vector<std::vector<float>> fliter_to;
  float f_v;
  int f_dimen; 
  int k;

  if (std::getline(file_filter, line)) {
    std::istringstream iss(line);
    iss >> f_dimen;  // Read the first element and assign it to A
  }

  if (std::getline(file_filter, line)) {
    std::istringstream iss(line);
    iss >> k;  // Read the first element and assign it to A
  }

  while (std::getline(file_filter, line)) {
        std::vector<float> fliter_v;
        std::istringstream iss(line);
        // Extract each floating-point number separated by whitespace
        while (iss >> f_v) {
            fliter_v.push_back(f_v);
        }

        // Add the row to the data vector
        fliter_to.push_back(fliter_v);
  }

  //file_filter.close();


  /*file_filter >> k;

  for (int i=0; i < k; i++){
    std::vector<float> fliter_v;
    for (int j=0; j < 3; j++){
      for (int z=0; z < 3; z++){
        file_filter >> f_v;
        fliter_v.push_back(f_v);
      }
    }
    fliter_to.push_back(fliter_v);
  }
  */
  int pad_amount = f_dimen - 1;
  int h_p_am = pad_amount / 2;


  if (std::getline(file_in, line)) {
      std::istringstream iss(line);
      iss >> h;  // Read the first element and assign it to A
  }

  if (std::getline(file_in, line)) {
      std::istringstream iss(line);
      iss >> w;  // Read the first element and assign it to A
  }

  if (std::getline(file_in, line)) {
      std::istringstream iss(line);
      iss >> n;  // Read the first element and assign it to A
  }

  std::vector<std::vector<float>> input_to;
  
  while (std::getline(file_in, line)) {
    std::vector<float> input_v;
    std::istringstream iss(line);
    float v_iN;

    // Extract each floating-point number separated by whitespace
    while (iss >> v_iN) {
        input_v.push_back(v_iN);
    }

    // Add the row to the data vector
    input_to.push_back(input_v);
  }

  //file_in.close();


	/*for(int i=0; i<3; i++)
	{
		file_in >> hw[i];
	}

  
  int h = hw[0];
  int w = hw[1];
  int n = hw[2]; 

  //int in_total = h * w;

	
  float v_iN;
	
  for (int i=0; i < n; i++){
    std::vector<float> input_v;
    for (int j=0; j < h; j++){
      for (int z=0; z < w; z++){
        file_filter >> v_iN;
        input_v.push_back(v_iN);
        numElements++;
      }
    }
    input_to.push_back(input_v);
  }
  */

	//file_in.close();

  int padded_width = w + pad_amount; 
  int padded_height = h + pad_amount;
  
  float* padded_image_to = (float*)malloc(n * padded_width * padded_height * sizeof(float));
  float* padded_image = (float*)malloc(padded_width * padded_height * sizeof(float));
  
  for (int z = 0; z < n; z++){
    
    for (int i = 0; i < padded_height; i++) {
      for (int j = 0; j < padded_width; j++) {
          padded_image[i * padded_width + j] = 0;
      }
    }

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
          padded_image[(y + h_p_am) * padded_width + (x + h_p_am)] = input_to[z*h + y][x];
      }
    }

    for (int y = 0; y < padded_height; y++) {
      for (int x = 0; x < padded_width; x++) {
          padded_image_to[z*(padded_width*padded_height) + y*padded_width + x] = padded_image[y*padded_width + x];
      }
    }

  }
  //file_filter.open("filter0.txt", std::ifstream::in);

    /*for(int i = 0; i < 3*k; i++){
      for(int j = 0; j < 3; j++){
          printf("%0.3f\n", fliter_to[i][j]);
      }
    }*/

  /*
  for(int j = 0; j < n * padded_width * padded_height+196; j++){
          printf("%0.3f\n", padded_image_to[j]);
      }
  */

	//file_filter.close();  

  int pad_total_orig = padded_width * padded_height;
  int pad_total = padded_width * padded_height * n;
	size_t size = w*h*n*k * sizeof(float);
  size_t pad_size = pad_total * sizeof(float);
	float *A_in= (float *)malloc(pad_size);
	float *A_out= (float *)malloc(size);
  int filter_total = f_dimen * f_dimen * k;
  size_t filter_size = filter_total * sizeof(float);
  float *A_fil= (float *)malloc(filter_size);

  for(int i=0; i<pad_total; i++) A_in[i] = padded_image_to[i];

  //for(int i=0; i<filter_total; i++) A_fil[i] = fliter_to[i];

  
  for(int i=0; i < k*f_dimen; i++){
    for(int j=0; j < f_dimen; j++){
      A_fil[i*f_dimen + j] = fliter_to[i][j];
    }
  } 

  
  float *d_input = NULL;
    err = cudaMalloc((void **)&d_input, pad_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  float *d_filter = NULL;
    err = cudaMalloc((void **)&d_filter, filter_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  float *d_output = NULL;
  err = cudaMalloc((void **)&d_output, size);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device d_output (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);



  err = cudaMemcpy(d_input, A_in, pad_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy input array d_input from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_filter, A_fil, filter_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy d_filter from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid =(pad_total + threadsPerBlock - 1) / threadsPerBlock;

  cudaEventRecord(start);
  
  conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter, d_output, f_dimen, h, w, padded_width, numElements, pad_total, pad_total_orig, n, k, padded_height);
  
  
  
  cudaEventRecord(stop);
  
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch conv2d_kernel kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(A_out, d_output, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy out array d_output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);


  /*std::ofstream o_file("output2.txt");
  if (!o_file.is_open()) {
      std::cerr << "Error opening file for writing." << std::endl;
      return 1;
  }

  // Write the array to the file in the desired format
  for (int i = 0; i < w*h*n*k; ++i) {
      o_file << A_out[i] << " ";
      if ((i + 1) % w == 0) { // After every 10 elements, insert a newline
          o_file << std::endl;
      }
  }

  // Close the file
  o_file.close();*/


  for(int i = 0; i < w*h*n*k; i++){
    printf("%0.3f　", A_out[i]);
    if ((i + 1) % w == 0) { 
         printf("\n");
      }
  }


   /*std::ifstream file("output2.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Use std::noskipws to consider whitespace characters during reading
    char ch;
    while (file >> std::noskipws >> ch) {
        std::cout << ch;
    }

    file.close();*/

  //float output_arry[n][k][h][w];
  
  /*float output_arry[h*k*n][w];


  for(int i = 0; i < h*k*n; i++){
    for(int j = 0; j < w; j++){
      output_arry[i][j] = A_out[i*w+j];
    }
  }*/

  
  
  
  //A*k*w*h + Z*h*w + i*w + j
  
 /* int counter = 0;
  for(int A = 0; A < n; A++){
    for(int Z = 0; Z < k; Z++){
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
          output_arry[A][Z][i][j] = A_out[counter];
          counter++;
        }
      } 
    }
  }*/


  
//printf("%f\n", milliseconds);

  /*for (int i = 0; i < w*h*n*k; ++i)
    {
        printf("%0.3f\n", A_out[i]);
    }
  */

     
    /*
      for(int i = 0; i < h*k*n; i++){
        for(int j = 0; j < w; j++){
          printf("%0.3f　", output_arry[i][j]);
        }
        printf("\n");
      }
    */

    /*  for(int A = 0; A < n; A++){
    for(int Z = 0; Z < k; Z++){
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
           printf("%0.3f　", output_arry[A][Z][i][j]);
        }
        printf("\n");
      } 
    }
  }*/



  err = cudaFree(d_input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_filter);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  free(A_in);
  free(A_out);
  free(A_fil);

  err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  return 0;
}