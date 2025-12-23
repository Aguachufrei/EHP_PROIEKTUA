// KONPILATZEKO - PARA COMPILAR: (C: -lm) (CUDA: -arch=sm_61)
// EXEC: analogy embeddings.dat dictionary.dat
// Ej., king – man + woman = queen

#include <cmath>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 10000     // Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE 50  	     // Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM 25		     // Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario
#define BLTAM 128

// Hitz baten indizea kalkulatzeko funtzioa
// Función para calcular el indice de una palabra
int word2ind(char* word, char** dictionary, int numwords) {
	for (int i = 0; i < numwords; i++) {
		if (strcmp(word, dictionary[i]) == 0) {
			return i;
		}
	}
	return -1;  // if the word is not found
}

// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
double dot_product(float* a, float* b, int size) {
	double result = 0;
	for (int i = 0; i < size; i++) {
		result += a[i] * b[i];
	}
	return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
float magnitude(float* vec, int size) {
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += vec[i] * vec[i];
	}
	return sqrt(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
float cosine_similarity(float* vec1, float* vec2, int size) {
	float mag1, mag2;

	mag1 = magnitude(vec1, size);
	mag2 = magnitude(vec2, size);
	return dot_product(vec1, vec2, size) / (mag1 * mag2);
}

// Analogia kalkulatzeko funtzioa
// Función para calcular la analogía
__global__ void perform_analogy(float *words, int idx1, int idx2, int idx3, float *result_vector) {
	/*****************************************************************
	  result_vector = word1_vector - word2_vector + word3_vector
	  OSATZEKO - PARA COMPLETAR
	 *****************************************************************/
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float *vector1 = &words[idx1 * EMB_SIZE];
	float *vector2 = &words[idx2 * EMB_SIZE];
	float *vector3 = &words[idx3 * EMB_SIZE];
	result_vector[i] = vector1[i] - vector2[i] + vector3[i];
}
/*
// Lortutako bektorearen gertukoen hitza bilatzeko funtzioa
// Función para encontrar la palabra más cercana al vector resultante
void find_closest_word(float *result_vector, float *words, int numwords, int idx1, int idx2, int idx3, int *closest_word_idx, float *max_similarity) {
	*max_similarity = -1.0;
	*closest_word_idx = -1;
	float result_mag = magnitude(result_vector, EMB_SIZE);
	for (int i = 0; i < numwords; i++) {
		if (i == idx1 || i == idx2 || i == idx3) {
			continue;
		}

		float *current_word_vector = &words[i * EMB_SIZE];
		float current_mag = magnitude(current_word_vector, EMB_SIZE);

		float similarity;

		//Ezin da zati 0 egin, hori ez egiteko hurrengo konprobazioa egiten da.
		if (result_mag > 0.0 && current_mag > 0.0) {
			similarity = cosine_similarity(result_vector, current_word_vector, numwords);
		} else {
			similarity = -2.0;
		}

		if (similarity > *max_similarity) {
			*max_similarity = similarity;
			*closest_word_idx = i;
		}
	}
}
*/


__global__ void magnitude_array(float *d_words, int size, float *d_magnitude){
	extern __shared__ float temp[];
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	
	//Biderketa kalkulatzen da bakarrik matrizearen barruan bagaude.
	float balioa = 0.0;
	if(idx<size){
		balioa = d_words[idx] * d_words[idx];
	}
	temp[tid] = balioa;	
	__syncthreads();

	//Redukzioa
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
    		if ((tid % (2*stride)) == 0 && tid + stride < blockDim.x){
        		temp[tid] += temp[tid + stride];
		}
    		__syncthreads();
	}
	
	//Idazketa
	if (tid == 0) {
    		d_magnitude[blockIdx.x] = sqrt(temp[0]);
	}


}


__global__ void find_similarities(
	float *result_vector, 
	float *words, 
	float *magnitudes, 
	float *similarities, 
	int size, 
	int numwords, 
	int idx1, int idx2, int idx3, 
	float result_magnitude
){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= numwords || idx == idx1 || idx == idx2 || idx == idx3)return;
	float dotp = 0.0;
	float *current_word_vector = &words[idx * size];

	for (int i = 0; i<size; i++){
		dotp += result_vector[i] * current_word_vector[i];
	}
	
	float current_magnitude = magnitudes[idx];

	if(result_magnitude > 0 && current_magnitude > 0){
		similarities[idx] = dotp / (result_magnitude*current_magnitude);
	} else {
		similarities[idx] = -2.0;
	}
}


void find_closest_word(
		float *result_vector, 
		float *words, 
		int numwords, 
		int idx1, int idx2, int idx3, 
		int *closest_word_idx, 
		float *max_similarity
) {

	float *d_words, *d_magnitudes, *d_similarities, *d_result_vector;

	cudaMalloc(&d_words, numwords * EMB_SIZE * sizeof(float));
	cudaMalloc(&d_magnitudes, numwords * sizeof(float));
	cudaMalloc(&d_similarities, numwords * sizeof(float));
	cudaMalloc(&d_result_vector, EMB_SIZE * sizeof(float));
	cudaMemcpy(d_words, words, EMB_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result_vector, result_vector, EMB_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	int blkop = numwords;

	//magnitudeak pre-kalkulatzen dira eta d_magnitudes-en uzten da
	magnitude_array<<<blkop, BLTAM, BLTAM * sizeof(float)>>>(d_words, EMB_SIZE, d_magnitudes);
	cudaDeviceSynchronize();
ir
	float result_magnitude = magnitude(result_vector, EMB_SIZE); 

	find_similarities<<<blkop, BLTAM>>>(
		d_result_vector, 
		d_words, 
		d_magnitudes, 
		d_similarities, 
		EMB_SIZE, 
		numwords, 
		idx1, idx2, idx3, 
		result_magnitude
	);

	float *similarities = (float*)malloc(numwords * sizeof(float));
	
	cudaMemcpy(similarities, d_similarities, numwords * sizeof(float), cudaMemcpyDeviceToHost);

	*max_similarity = -1.0;
	*closest_word_idx = -1;

	for (int i = 0; i < numwords; i++) {
		if (similarities[i] > *max_similarity) {
			*max_similarity = similarities[i];
			*closest_word_idx = i;
		}
	}
	free(similarities);
	cudaFree(d_words);
	cudaFree(d_magnitudes);
	cudaFree(d_similarities);
	cudaFree(d_result_vector);
}


int main(int argc, char *argv[])
{
	int		i, j, numwords, idx1, idx2, idx3;
	int 	closest_word_idx;
	float	max_similarity;
	float 	*words; //numwords * EMB_SIZE
	FILE    	*f1, *f2;
	char 	**dictionary; //numwords
	char	target_word1[TAM], target_word2[TAM], target_word3[TAM];
	float	*result_vector; //EMB_SIZE
	float	*sim_cosine; //numwords

	struct timespec  t0, t1;
	double tej;


	if (argc < 3) {
		printf("Deia: analogia embedding_fitx hiztegi_fitx\n");
		exit (-1);;
	}


	// Irakurri datuak sarrea-fitxategietatik
	// ======================================
	f1 = fopen (argv[1], "r");
	if (f1 == NULL) {
		printf ("Errorea %s fitxategia irekitzean\n", argv[1]);
		exit (-1);
	}

	f2 = fopen (argv[2], "r");
	if (f1 == NULL) {
		printf ("Errorea %s fitxategia irekitzean\n", argv[2]);
		exit (-1);
	}


	fscanf (f1, "%d", &numwords);	       // prozesatu behar den hitz kopurua fitxategitik jaso
	if (argc == 4) numwords = atoi (argv[3]);   // 3. parametroa = prozesatu behar diren hitzen kopurua
	printf ("numwords = %d\n", numwords);

	words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
	dictionary = (char**)malloc (numwords*sizeof(char*));
	for (i=0; i<numwords;i++){
		dictionary[i] = (char*)malloc(TAM*sizeof(char));
	}
	sim_cosine = (float*)malloc (numwords*sizeof(float));
	result_vector = (float*)malloc (EMB_SIZE*sizeof(float));

	for (i=0; i<numwords; i++) {
		fscanf (f2, "%s", dictionary[i]);
		for (j=0; j<EMB_SIZE; j++) {
			fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
		}
	}
	printf("Sartu analogoak diren bi hitzak eta analogia bilatu nahi diozun hitza: \n");
	printf("Introduce las dos palabras analogas y la palabra a la que le quieres buscar la analogia: \n");
	scanf ("%s %s %s",target_word1, target_word2, target_word3);

	/*********************************************************************
	  OSATZEKO - PARA COMPLETAR
	  Sartutako hitzen indizeak kalkulatu (idx1, idx2 & idx3) word2ind funtzioa erabilita
	  Calcular los indices de las palabras introducidas (idx1, idx2 & idx3) con la funcion word2ind
	 **********************************************************************/
	idx1 = word2ind(target_word1, dictionary, numwords);
	idx2 = word2ind(target_word2, dictionary, numwords);
	idx3 = word2ind(target_word3, dictionary, numwords);
	if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
		printf("Errorea: Ez daude hitz guztiak hiztegian / No se encontraron todas las palabras en el vocabulario.\n");
		return -1;
	}

	clock_gettime (CLOCK_REALTIME, &t0);
	/***************************************************/
	//    OSATZEKO - PARA COMPLETAR
	//     1. call perform_analogy function
	//     2. call find_closest_word function
	/***************************************************/


	int blkop = EMB_SIZE/BLTAM+((EMB_SIZE%BLTAM)==0?0:1);
	float *d_words, *d_result_vector;
	cudaMalloc(&d_words, numwords*EMB_SIZE*sizeof(float));
	cudaMalloc(&d_result_vector, EMB_SIZE*sizeof(float));
	cudaMemcpy(d_words, words, numwords*EMB_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	perform_analogy<<<blkop, BLTAM>>>(d_words, idx1, idx2, idx3, d_result_vector);

	cudaMemcpy(result_vector, d_result_vector, EMB_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_words);
	cudaFree(d_result_vector);


	//find_closest_word
	find_closest_word(result_vector, words, numwords, idx1, idx2, idx3, &closest_word_idx, &max_similarity);

	clock_gettime (CLOCK_REALTIME, &t1);

	if (closest_word_idx != -1) {
		printf("\nClosest_word: %s (%d), sim = %f \n", dictionary[closest_word_idx],closest_word_idx, max_similarity);
	} else printf("No close word found.\n");


	tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
	printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

	fclose (f1);
	fclose (f2);

	free(words);
	free(sim_cosine);
	free(result_vector);
	for (i=0; i<numwords;i++) free (dictionary[i]);
	free(dictionary);

	return 0;
}
