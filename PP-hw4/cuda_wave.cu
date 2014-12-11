/**********************************************************************
* DESCRIPTION:
*   Serial Concurrent Wave Equation - C Version
*   This program implements the concurrent wave equation
*********************************************************************/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

#define cudaFn(fn,...) { \
	cudaError_t __cudaStatus = fn(__VA_ARGS__); \
	if (__cudaStatus != cudaSuccess) { \
		fprintf(stderr, "Failed when calling cuda function \"%s\"!\n%s\n", #fn, cudaGetErrorString(__cudaStatus)); \
		exit(1); \
	} \
}
#define kernelFn(fn,block_count,thread_count,...) { \
	fn<<<block_count, thread_count>>>(__VA_ARGS__); \
	cudaError_t __cudaStatus = cudaGetLastError(); \
	if (__cudaStatus != cudaSuccess) { \
		fprintf(stderr, "%s launch failed: %s\n", #fn, cudaGetErrorString(__cudaStatus)); \
		exit(1); \
	} \
}

float result[MAXPOINTS + 2];

/**********************************************************************
*	Checks input values from parameters
*********************************************************************/
__host__ void check_param(int *tpoints, int *nsteps) {
	char tchar[20];

	/* check number of points, number of iterations */
	while ((*tpoints < MINPOINTS) || (*tpoints > MAXPOINTS)) {
		printf("Enter number of points along vibrating string [%d-%d]: "
			   , MINPOINTS, MAXPOINTS);
		scanf("%s", tchar);
		*tpoints = atoi(tchar);
		if ((*tpoints < MINPOINTS) || (*tpoints > MAXPOINTS))
			printf("Invalid. Please enter value between %d and %d\n",
			MINPOINTS, MAXPOINTS);
	}
	while ((*nsteps < 1) || (*nsteps > MAXSTEPS)) {
		printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
		scanf("%s", tchar);
		*nsteps = atoi(tchar);
		if ((*nsteps < 1) || (*nsteps > MAXSTEPS))
			printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
	}

	printf("Using points = %d, steps = %d\n", *tpoints, *nsteps);

}

/**********************************************************************
*      Calculate new values using wave equation
*********************************************************************/
__device__ float do_math(float values, float oldval) {
	float dtime, c, dx, tau, sqtau;

	dtime = 0.3;
	c = 1.0;
	dx = 1.0;
	tau = (c * dtime / dx);
	sqtau = tau * tau;
	return (2.0 * values) - oldval + (sqtau *  (-2.0)*values);
}

__global__ void update(float *values, int tpoints, int nsteps) {
	/**********************************************************************
	*     Initialize points on line
	*********************************************************************/
	int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	float x, fac;
	float nowval,	/* values at time t */ 
		oldval,		/* values at time (t-dt) */
		newval;		/* values at time (t+dt) */

	/* Calculate initial values based on sine curve */
	fac = 2.0 * PI;
	x = (-1.0 + idx) / (float) (tpoints - 1);

	nowval = sin(fac * x);

	/* Initialize old values array */
	oldval = nowval;




	/**********************************************************************
	*     Update all values along line a specified number of times
	*********************************************************************/
	int i;

	/* Update values for each time step */
#pragma unroll 1000
	for (i = 0; i < nsteps; i++) {
		/* Update points along line for this time step */
		newval = do_math(nowval, oldval);

		/* Update old values with new values */
		oldval = nowval;
		nowval = newval;
	}

	/* global endpoints */
	if ((idx == 1) || (idx == tpoints))
		nowval = 0.0;

	values[idx] = nowval;
}

/**********************************************************************
*     Print final results
*********************************************************************/
__host__ void printfinal(float values[], int tpoints) {
	int i;

	for (i = 1; i <= tpoints; i++) {
		printf("%6.4f ", values[i]);
		if (i % 10 == 0)
			printf("\n");
	}
}

/**********************************************************************
*	Main program
*********************************************************************/
__host__ int main(int argc, char *argv[]) {
	int nsteps,                 	/* number of time steps */
		tpoints; 	     		/* total points along string */
	float *values;

	if (argc != 3) {
		printf("Usage: %s <number_of_points> <number_of_time_steps>\n", argv[0]);
		return 1;
	}
	sscanf(argv[1], "%d", &tpoints);
	sscanf(argv[2], "%d", &nsteps);
	check_param(&tpoints, &nsteps);

	int blockCount = tpoints / 1024 + 1;
	int arySize = (blockCount * 1024 + 1) * sizeof(float);
	cudaFn(cudaMalloc, &values, arySize);

	printf("Initializing points on the line...\n");
	printf("Updating all points for all time steps...\n");
	kernelFn(update, blockCount, 1024, values, tpoints, nsteps);

	printf("Printing final results...\n");
	cudaFn(cudaMemcpy, result, values, arySize, cudaMemcpyDeviceToHost);
	printfinal(result, tpoints);
	printf("\nDone.\n\n");

	cudaFn(cudaFree, values);
	cudaFn(cudaDeviceReset);
	return 0;
}
