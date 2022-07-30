
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>  
#include <algorithm>

#include <string>
#include <math.h>
using namespace std;

cudaError_t labelWithCuda(float(*vertices)[2], int(*triangles)[3], int(*neighbors)[3], int* leArr, int(*feArr)[3], bool* seedArr, const int arraysize, const int verticesSize);

__device__ float pDistance(float *a, float *b) {
    float vertex1_x = a[0];
    float vertex1_y = a[1];
    float vertex2_x = b[0];
    float vertex2_y = b[1];

    float length = powf((powf((vertex2_x - vertex1_x), 2) + powf((vertex2_y - vertex1_y), 2)), 0.5);
    return length;
}

__device__ int getLongestIndex(float(*dev_vertices)[2], int(*dev_triangles)[3], int tindex) {
    int* t_vertices = dev_triangles[tindex];
    float *v1, *v2, *v3;
    v1 = dev_vertices[t_vertices[0]];
    v2 = dev_vertices[t_vertices[1]];
    v3 = dev_vertices[t_vertices[2]];

    float distances[] = { pDistance(v2, v3), pDistance(v3, v1), pDistance(v1, v2) };
    if (distances[0]> distances[1]){
        if (distances[0] > distances[2]) {
            return 0;
        }
        else {
            return 2;
        }
    }
    else {
        if (distances[1] > distances[2])
            return 1;
        else
            return 2;
    }

}

__device__ bool areEqual(float *a, float *b) {
    if (fabs(a[0] - b[0]) <= 10e-15 * fmaxf(fmaxf(a[0], b[0]), 1.0)
        && fabs(a[1] - b[1]) <= 10e-15 * fmaxf(fmaxf(a[1], b[1]), 1.0)) {
        return 1;
    }
    return 0;
}

__device__ int oppIdx(float(*dev_vertices)[2], int(*dev_triangles)[3], int(*dev_neighbors)[3], int tindex, int nindex) {
    float* a = dev_vertices[dev_triangles[tindex][(nindex + 1) % 3]];
    float* b = dev_vertices[dev_triangles[tindex][(nindex + 2) % 3]];

    int t_n_idx = dev_neighbors[tindex][nindex];
    int *n_idx_ver = dev_triangles[t_n_idx];
    float* actualVertex;
    
    
    for (int i = 0; i < 3; i++){
        actualVertex = dev_vertices[n_idx_ver[i]];
        if (!areEqual(a, actualVertex ) && !areEqual(actualVertex , b)) {
            //printf("%f %f\n", a[0], actualVertex[0]);
            return i;
        }
    }

    return -1000000;
    
}

__global__ void labelKernel(float (*dev_vertices)[2], int (*dev_triangles)[3], 
    int(*dev_neighbors)[3], int *dev_leArr, bool *dev_seedArr, int (*dev_feArr)[3], const int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size - 10) {
        int leidx = getLongestIndex(dev_vertices, dev_triangles, i);
        int neighbor_i = dev_neighbors[i][leidx];

        dev_leArr[i] = leidx;

        //Seed Labeling
        if (neighbor_i != -1) { // If neighbor exists 
            int oppIndex = oppIdx(dev_vertices, dev_triangles, dev_neighbors, i, leidx);
            int nleidx = getLongestIndex(dev_vertices, dev_triangles, neighbor_i);
            if (oppIndex == nleidx) {
                dev_seedArr[i] = 1;
            }
            else {
                dev_seedArr[i] = 0;
            }

        }
        else {
            dev_seedArr[i] = 1;
        }


        //Frontier Labeling
        //int oppIndex = oppIdx(dev_vertices, dev_triangles, dev_neighbors, i, leidx);
        if (neighbor_i == -1) {
            dev_feArr[i][leidx] = -1;
        }
        else {
            dev_feArr[i][leidx] = 0;
        }

        int leidx2 = (leidx + 1) % 3;
        int leidx3 = (leidx + 2) % 3;
        int neighbor_i2 = dev_neighbors[i][leidx2];
        int neighbor_i3 = dev_neighbors[i][leidx3];
        if (neighbor_i2 != -1) {
            int oppindex2 = oppIdx(dev_vertices, dev_triangles, dev_neighbors, i, leidx2);
            int nleidx2 = getLongestIndex(dev_vertices, dev_triangles, neighbor_i2);
            if (oppindex2 == nleidx2) { //if it is the longest edge of neighbor
                dev_feArr[i][leidx2] = 0;
            }
            else {
                dev_feArr[i][leidx2] = 1;
            }
        }
        else {
            dev_feArr[i][leidx2] = 1;
        }

        if (neighbor_i3 != -1) {
            int oppindex3 = oppIdx(dev_vertices, dev_triangles, dev_neighbors, i, leidx3);
            int nleidx3 = getLongestIndex(dev_vertices, dev_triangles, neighbor_i3);
            if (oppindex3 == nleidx3) { //if it is the longest edge of neighbor
                dev_feArr[i][leidx3] = 0;
            }
            else {
                dev_feArr[i][leidx3] = 1;
            }
        }
        else {
            dev_feArr[i][leidx3] = 1;
        }
    }
}


int main()
{
    //Load mesh input file


    string line;
    string operation;
    ifstream myfile("./examples/sample1000.txt");

    float(*vertices)[2];
    int(*triangles)[3];
    int(*neighbors)[3];
    string v1, v2;
    string t1, t2, t3;
    string n1, n2, n3;
    string verticesNumber, trianglesNumber, neighborsNumber;
    int vcount = 0, tcount = 0, ncount = 0;
    if (myfile.is_open())
    {

        while (getline(myfile, line))
        {
            stringstream s(line);
            s >> operation;
            if (operation == "Vertices") {

                s >> verticesNumber;
                vertices = new float[stoi(verticesNumber)][2];
            }
            else if (operation == "Triangles") {

                s >> trianglesNumber;

                triangles = new int[stoi(trianglesNumber) + 10][3];
                neighbors = new int[stoi(trianglesNumber) + 10][3];

            }
            else if (operation == "v") {

                s >> v1 >> v2;
                vertices[vcount][0] = stof(v1);
                vertices[vcount][1] = stof(v2);
                vcount++;
            }
            else if (operation == "t") {

                s >> t1 >> t2 >> t3;
                triangles[tcount][0] = stoi(t1);
                triangles[tcount][1] = stoi(t2);
                triangles[tcount][2] = stoi(t3);
                tcount++;
            }

            else if (operation == "n") {

                s >> n1 >> n2 >> n3;
                neighbors[ncount][0] = stoi(n1);
                neighbors[ncount][1] = stoi(n2);
                neighbors[ncount][2] = stoi(n3);
                ncount++;
            }

        }
        myfile.close();
    }

    else cout << "Unable to open file";

    const int arraysize = stoi(trianglesNumber) + 10;
    const int verticesSize = stoi(verticesNumber);
    int* longestEdgeArray = new int[arraysize] {0};
    bool* seedEdgeArray = new bool[arraysize] {0};
    int(* frontierEdgeArray)[3] = new int[arraysize][3];

    // Label phase in parallel.
    cudaError_t cudaStatus = labelWithCuda(vertices, triangles, neighbors, longestEdgeArray, 
        frontierEdgeArray, seedEdgeArray, arraysize, verticesSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    ofstream outfile;
    outfile.open ("label_output.txt");

    outfile << "LongestEdges " << arraysize - 10 << endl << endl;

    for (int t = 0; t < arraysize - 10; t++) {
        outfile << "le " << longestEdgeArray[t] << endl;
    }

    cout << endl;

    outfile << "Seed " << arraysize - 10 << endl << endl;

    for (int t = 0; t < arraysize - 10; t++) {
        outfile << "s " << seedEdgeArray[t] << endl;
    }

    outfile << "FrontierEdges " << arraysize - 10 << endl << endl;
    

    for (int t = 0; t < arraysize - 10; t++) {
        outfile << "fe " << frontierEdgeArray[t][0] << " " << frontierEdgeArray[t][1] << " "
            << frontierEdgeArray[t][2] << endl;
    }

    outfile.close();


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


//Helper function for using CUDA to label triangles in parallel
cudaError_t labelWithCuda(float(*vertices)[2], int(*triangles)[3], int(*neighbors)[3], int* leArr, 
    int(* feArr)[3], bool* seedArr, const int size, const int verticesSize) {
    float(*dev_vertices)[2] = 0;
    int(*dev_triangles)[3] = 0;
    int(*dev_neighbors)[3] = 0;
    int* dev_leArr = 0;
    bool* dev_seedArr = 0;
    int(*dev_feArr)[3] = 0;

    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_vertices, verticesSize * sizeof(float) * 2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_triangles, size * sizeof(int) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_neighbors, size * sizeof(int) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_leArr, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_feArr, size * sizeof(int) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_seedArr, size * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_vertices, vertices, verticesSize * sizeof(float) * 2, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_triangles, triangles, size * sizeof(int) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_neighbors, neighbors, size * sizeof(int) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

   

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    unsigned t0, t1;
    t0 = clock();

    // Launch a kernel on the GPU with one thread for each element.
    labelKernel << <blocksPerGrid, threadsPerBlock >> > (dev_vertices, dev_triangles, dev_neighbors, dev_leArr,
        dev_seedArr, dev_feArr, size);

    

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    t1 = clock();
    double time = (double(t1 - t0) / CLOCKS_PER_SEC);
    std::cout << "Execution Time: " << std::fixed
        << std::setprecision(5) << time << " s in CUDA" << '\n';

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(leArr, dev_leArr, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(seedArr, dev_seedArr, size * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(feArr, dev_feArr, size * sizeof(int) * 3, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
    cudaFree(dev_vertices);
    cudaFree(dev_triangles);
    cudaFree(dev_neighbors);
    cudaFree(dev_feArr);
    cudaFree(dev_leArr);

    return cudaStatus;

    }






  
