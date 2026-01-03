
// EXEKUTATZEKO: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]    // numwords: matrize txikiekin probak egiteko 

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>


#define VOCAB_SIZE  10000	// Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE    50		// Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM         25		// Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario
#define MAX_ITER    1000	// konbergentzia: iterazio kopuru maximoa -- Convergencia: num maximo de iteraciones
#define K	    20		// kluster kopurua -- numero de clusters
#define DELTA       0.5		// konbergentzia (cvi) -- convergencia (cvi)
#define NUMCLUSTERSMAX 100	// cluster kopuru maximoa -- numero máximo de clusters


struct clusterinfo	 // clusterrei buruzko informazioa -- informacion de los clusters
{
	int  elements[VOCAB_SIZE];	// osagaiak -- elementos
	int  number;			// osagai kopurua -- número de elementos
};


// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
float dot_product(float* a, float* b, int dim) {
	float result = 0;
	for (int i = 0; i < dim; i++) {
		result += a[i] * b[i];
	}
	return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
float magnitude(float* vec, int dim) {
	float sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += vec[i] * vec[i];
	}
	return sqrt(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
float cosine_similarity(float* vec1, float* vec2, int dim) {
	float mag1, mag2;

	mag1 = magnitude(vec1, dim);
	mag2 = magnitude(vec2, dim);
	if (mag1 == 0 || mag2 == 0) return 0; // Bektoreren bat 0 bada -- Si alguno de los vectores es nulo: cosine_similarity = 0
	else return dot_product(vec1, vec2, dim) / (mag1 * mag2);
}

//global funtzioak void motakoak dira eta ezin dute balioak itzuli, hau keneletik deitu daiteke eta ez da memcpy behar teorian.
__device__ float cosine_similarity_device(float *a, float *b, float norm_a, float norm_b, int dim) {
	float dot = 0.0;
	for (int i = 0; i < dim; i++) {
		dot += a[i] * b[i];
	}
	return dot / (norm_a * norm_b);
}

// Distantzia euklidearra: bi hitzen kenketa ber bi, eta atera erro karratua
// Distancia euclidea: raiz cuadrada de la resta de dos palabras elevada al cuadrado
// Adi: double
double word_distance (float *word1, float *word2)
{
	/****************************************************************************************
	  OSATZEKO - PARA COMPLETAR
	 ****************************************************************************************/
	double dist_sq = 0.0;
	for (int i = 0; i < EMB_SIZE; i++) {
		double diff = (double)word1[i] - (double)word2[i];
		dist_sq += diff * diff;
	}
	return sqrt(dist_sq);
}

// Zentroideen hasierako balioak ausaz -- Inicializar centroides aleatoriamente
void initialize_centroids(float *words, float *centroids, int n, int numclusters, int dim) {
	int i, j, random_index;
	for (i = 0; i < numclusters; i++) {
		random_index = rand() % n;
		for (j = 0; j < dim; j++) {
			centroids[i*dim+j] = words[random_index*dim+j];
		}
	}
}

// Zentroideak eguneratu -- Actualizar centroides
void update_centroids(float *words, float *centroids, int *wordcent, int numwords, int numclusters, int dim, int *cluster_sizes) {

	int i, j, cluster;

	for (int i = 0; i < numclusters; i++) {
		cluster_sizes[i]=0;
		for (int j = 0; j < dim; j++) {
			centroids[i*dim+j] = 0.0; // Zentroideak berrasieratu -- Reinicia los centroides
		}
	}

	for (i = 0; i < numwords; i++) {
		cluster = wordcent[i];
		cluster_sizes[cluster]++;
		for (j = 0; j < dim; j++) {
			centroids[cluster*dim+j] += words[i*dim+j];
		}
	}

	for (i = 0; i < numclusters; i++) {
		if (cluster_sizes[i] > 0) {
			for (j = 0; j < dim; j++) {
				centroids[i * dim + j] = centroids[i * dim + j] / cluster_sizes[i];
			}
		}
	}
}

//zentroideak eta clusterraren tamainak hasieratu
__global__ void reset_centroids_kernel(float *centroids, int *cluster_sizes, int numclusters, int dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numclusters) {
		cluster_sizes[idx] = 0;
		for (int j = 0; j < dim; j++) {
			centroids[idx * dim + j] = 0.0f;
		}
	}
}

__global__ void centroid_kernel(float *words, float *centroids, int *wordcent, 
		int *cluster_sizes, int numwords, int dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numwords)return;
	int cluster = wordcent[idx];
	atomicAdd(&cluster_sizes[cluster], 1);
	for (int i = 0; i < dim; i++) {
		atomicAdd(&centroids[cluster * dim + i], words[idx * dim + i]);
	}
}

__global__ void normalice_kernel(float *centroids, int *cluster_sizes, int numclusters, int dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numclusters && cluster_sizes[idx] > 0) {
		for (int i = 0; i < dim; i++) {
			centroids[idx * dim + i] /= cluster_sizes[idx];
		}
	}
}



void update_centroids_host(float *words, float *centroids, int *wordcent, int numwords, int numclusters, int dim, int *cluster_sizes) {
	float *d_words, *d_centroids;
	int *d_wordcent, *d_cluster_sizes;

	cudaMalloc(&d_words, numwords * dim * sizeof(float));
	cudaMalloc(&d_centroids, numclusters * dim * sizeof(float));
	cudaMalloc(&d_wordcent, numwords * sizeof(int));
	cudaMalloc(&d_cluster_sizes, numclusters * sizeof(int));

	cudaMemcpy(d_words, words, numwords * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wordcent, wordcent, numwords * sizeof(int), cudaMemcpyHostToDevice);

	int bltam = 256;
	int blKopClusters = (numclusters + bltam - 1) / bltam;
	int blKopWords = (numwords + bltam - 1) / bltam;

	reset_centroids_kernel<<<blKopClusters, bltam>>>(d_centroids, d_cluster_sizes, numclusters, dim);

	centroid_kernel<<<blKopWords, bltam>>>(d_words, d_centroids, d_wordcent, d_cluster_sizes, numwords, dim);

	normalice_kernel<<<blKopClusters, bltam>>>(d_centroids, d_cluster_sizes, numclusters, dim);

	//cudaDeviceSynchronize();

	cudaMemcpy(centroids, d_centroids, numclusters * dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cluster_sizes, d_cluster_sizes, numclusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_words);
	cudaFree(d_centroids);
	cudaFree(d_wordcent);
	cudaFree(d_cluster_sizes);
}

// K-Means funtzio nagusia -- Función principal de K-Means
void k_means_calculate(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed) 
{  
	/****************************************************************************************    
	  OSATZEKO - PARA COMPLETAR
	  - Hitz bakoitzari cluster gertukoena esleitu cosine_similarity funtzioan oinarrituta
	  - Asignar cada palabra al cluster más cercano basandose en la función cosine_similarity       
	 ****************************************************************************************/
	int i, j;

	*changed = 0;

	for (i = 0; i < numwords; i++) {
		float max_similarity = -1;
		int closest_centroid = -1;
		float *word_vector = &words[i * dim];

		for (j = 0; j < numclusters; j++) {
			float *centroid_vector = &centroids[j * dim];

			float similarity = cosine_similarity(word_vector, centroid_vector, dim);

			if (similarity > max_similarity) {
				max_similarity = similarity;
				closest_centroid = j;
			}
		}

		if (wordcent[i] != closest_centroid) {
			wordcent[i] = closest_centroid;
			*changed = 1;
		}
	}
}


__global__ void k_means_calculate_kernel(
		float *words,
		int numwords,
		int dim,
		int numclusters,
		float *centroids,
		float *norm_words,
		float *norm_centroids,
		int *wordcent,
		int *changed)
{
	extern __shared__ float shared_centroids[]; // dim * numclusters

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadIdx.x; i < numclusters * dim; i += blockDim.x) {
		shared_centroids[i] = centroids[i];
	}
	__syncthreads();

	if (idx < numwords) {
		float *word_vec = &words[idx * dim];
		float max_similarity = -1.0;
		int closest = -1;

		for (int i = 0; i < numclusters; i++) {
			float *cent_vec = &shared_centroids[i * dim];
			float sim = cosine_similarity_device(word_vec, cent_vec, norm_words[idx], norm_centroids[i], dim);
			if (sim > max_similarity) {
				max_similarity = sim;
				closest = i;
			}
		}

		if (wordcent[idx] != closest) {
			wordcent[idx] = closest;
			atomicOr(changed, 1);
		}
	}
}


void k_means_calculate_host(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed){  
	float *d_words, *d_centroids, *d_pre_words, *d_pre_centroids;
	int *d_wordcent, *d_changed;

	cudaMalloc(&d_words, numwords * dim * sizeof(float));
	cudaMalloc(&d_centroids, numclusters * dim * sizeof(float));
	cudaMalloc(&d_wordcent, numwords * sizeof(int));
	cudaMalloc(&d_pre_words, numwords * sizeof(float));
	cudaMalloc(&d_pre_centroids, numclusters * sizeof(float));
	cudaMalloc(&d_changed, sizeof(int));

	cudaMemcpy(d_words, words, numwords * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids, centroids, numclusters * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wordcent, wordcent, numwords * sizeof(int), cudaMemcpyHostToDevice);

	float *pre_words = (float*)malloc(numwords * sizeof(float));
	float *pre_centroids = (float*)malloc(numclusters * sizeof(float));

	//prekalkulatu hitzen centroide gertuenak cpuan momentuz.	
	for (int i = 0; i < numwords; i++) {
		float sum = 0;
		for (int j = 0; j < dim; j++) sum += words[i*dim + j] * words[i*dim + j];
		pre_words[i] = sqrtf(sum);
	}

	for (int i = 0; i < numclusters; i++) {
		float sum = 0;
		for (int j = 0; j < dim; j++) sum += centroids[i*dim + j] * centroids[i*dim + j];
		pre_centroids[i] = sqrtf(sum);
	}

	cudaMemcpy(d_pre_words, pre_words, numwords * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pre_centroids, pre_centroids, numclusters * sizeof(float), cudaMemcpyHostToDevice);

	int blkop = 256;//aldatu daiteke
	int bltam = (numwords + blkop - 1) / blkop; //borobilketa 
	int size = numclusters * dim * sizeof(float);
	*changed = 0;
	cudaMemcpy(d_changed, changed, sizeof(int), cudaMemcpyHostToDevice);

	k_means_calculate_kernel<<<bltam, blkop, size>>>(
			d_words, numwords, dim, numclusters, d_centroids,
			d_pre_words, d_pre_centroids, d_wordcent, d_changed
			);
	cudaMemcpy(wordcent, d_wordcent, numwords * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_words);
	cudaFree(d_centroids);
	cudaFree(d_wordcent);
	cudaFree(d_pre_words);
	cudaFree(d_pre_centroids);
	cudaFree(d_changed);
	free(pre_words);
	free(pre_centroids);
}


double cluster_homogeneity(float *words, struct clusterinfo *members, int i, int numclusters, int number)
{
	/****************************************************************************************
	  OSATZEKO - PARA COMPLETAR
	  Kideen arteko distantzien batezbestekoa - Media de las distancias entre los elementos del cluster
	  Cluster bakoitzean, hitz bikote guztien arteko distantziak - En cada cluster, las distancias entre todos los pares de elementos
	  Adi, i-j neurtuta, ez da gero j-i neurtu behar  / Ojo, una vez calculado el par i-j no hay que calcular el j-i
	 ****************************************************************************************/
	double disbat = 0.0;
	for (int k = 0; k < number; k++) {
		int word1_index = members[i].elements[k];
		float *word1 = &words[word1_index * EMB_SIZE];

		for (int l = k + 1; l < number; l++) {
			int word2_index = members[i].elements[l];
			float *word2 = &words[word2_index * EMB_SIZE];

			disbat += word_distance(word1, word2);
		}
	}
	return disbat;
}

double centroid_homogeneity(float *centroids, int i, int numclusters)
{
	/****************************************************************************************
	  OSATZEKO - PARA COMPLETAR
	 ****************************************************************************************/
	double disbat = 0.0;
	float *centroid_i = &centroids[i * EMB_SIZE];
	for (int j = 0; j < numclusters; j++) {
		if (i != j) {
			float *centroid_j = &centroids[j * EMB_SIZE];
			disbat += word_distance(centroid_i, centroid_j);
		}
	}
	return disbat;
}


double validation (float *words, struct clusterinfo *members, float *centroids, int numclusters)
{
	int     i, number;
	float   cent_homog[numclusters];
	double  disbat, max, cvi;
	float   clust_homog[numclusters];	// multzo bakoitzeko trinkotasuna -- homogeneidad de cada cluster


	// Kalkulatu clusterren trinkotasuna -- Calcular la homogeneidad de los clusters
	// Cluster bakoitzean, hitz bikote guztien arteko distantzien batezbestekoa. Adi, i - j neurtuta, ez da gero j - i neurtu behar
	// En cada cluster las distancias entre todos los pares de palabras. Ojo, una vez calculado i - j, no hay que calcular el j - i

	for (i=0; i<numclusters; i++)
	{
		disbat = 0.0;
		number = members[i].number; 
		if (number > 1)     // min 2 members in the cluster
		{
			disbat = cluster_homogeneity(words, members, i, numclusters, number);
			clust_homog[i] = disbat/(number*(number-1)/2);	// zati bikote kopurua -- div num de parejas
		}
		else clust_homog[i] = 0;


		// Kalkulatu zentroideen trinkotasuna -- Calcular la homogeneidad de los centroides
		// clusterreko zentroidetik gainerako zentroideetarako batez besteko distantzia 
		// dist. media del centroide del cluster al resto de centroides

		disbat = centroid_homogeneity(centroids, i, numclusters);
		cent_homog[i] = disbat / (numclusters-1);	// 5 multzo badira, 4 distantzia batu dira -- si son 5 clusters, se han sumado 4 dist.
	}

	// cvi index
	/****************************************************************************************
	  OSATZEKO - PARA COMPLETAR
fmaxf: max of 2 floats --> maximoa kalkulatzeko -- para calcular el máximo
	 ****************************************************************************************/
	cvi = 0.0;
	for(int i=0; i<numclusters;i++){
		max=fmax(cent_homog[i], clust_homog[i]);
		cvi +=(cent_homog[i] - clust_homog[i])/max;
	}
	cvi=cvi/numclusters;
	return (cvi);
}


int main(int argc, char *argv[]) 
{
	int		i, j, numwords, k, iter, changed, end_classif;
	int		cluster, zenb, numclusters = 20;
	double	cvi, cvi_zaharra, dif;
	float	*words;
	FILE	*f1, *f2, *f3;
	char	**hiztegia;  
	int		*wordcent;

	struct clusterinfo  members[NUMCLUSTERSMAX];

	struct timespec  t0, t1;
	double tej;


	if (argc < 4) {
		printf("\nCall: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]\n\n");
		printf("\t(in) embeddings.dat and dictionary.dat\n");
		printf("\t(out) myclusters.dat\n");
		printf("\t(numwords optional) prozesatu nahi den hitz kopurua -- num de palabras a procesar\n\n");
		exit (-1);;
	}  

	// Irakurri datuak sarrea-fitxategietatik -- Leer los datos de los ficheros de entrada
	// =================================================================================== 

	f1 = fopen (argv[1], "r");
	if (f1 == NULL) {
		printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[1]);
		exit (-1);
	}

	f2 = fopen (argv[2], "r");
	if (f2 == NULL) {
		printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[2]);
		exit (-1);
	}

	fscanf (f1, "%d", &numwords);	       
	if (argc == 5) numwords = atoi (argv[4]);  
	printf ("numwords = %d\n", numwords);

	words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
	hiztegia = (char**)malloc (numwords*sizeof(char*));
	for (i=0; i<numwords;i++){
		hiztegia[i] = (char*)malloc(TAM*sizeof(char));
	}

	for (i=0; i<numwords; i++) {
		fscanf (f2, "%s", hiztegia[i]);
		for (j=0; j<EMB_SIZE; j++) {
			fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
		}
	}
	printf ("Embeddingak eta hiztegia irakurrita -- Embeddings y dicionario leidos\n");

	wordcent = (int *)malloc(numwords * sizeof(int));
	for (int i = 0; i < numwords; i++) wordcent[i] = -1;

	k = NUMCLUSTERSMAX;   // hasierako kluster kopurua (20) -- numero de clusters inicial
	end_classif = 0; 
	cvi_zaharra = -1;

	float *centroids = (float *)malloc(k * EMB_SIZE * sizeof(float));
	int *cluster_sizes = (int *)calloc(k, sizeof(int));



	/******************************************************************/
	// A. kmeans kalkulatu -- Calcular kmeans
	// =========================================================
	printf("K_means\n");
	clock_gettime (CLOCK_REALTIME, &t0);

	while (numclusters < NUMCLUSTERSMAX && end_classif == 0)
	{
		initialize_centroids(words, centroids, numwords, numclusters, EMB_SIZE);
		for (iter = 0; iter < MAX_ITER; iter++) {
			changed = 0;
			/****************************************************************************************
			  OSATZEKO - PARA COMPLETAR
			  deitu k_means_calculate funtzioari -- llamar a la función k_means_calculate
			 ****************************************************************************************/
			k_means_calculate_host(words, numwords, EMB_SIZE, numclusters, wordcent, centroids, &changed);
			//Paraleloan motelago egiten da
			//update_centroids_host(words, centroids, wordcent, numwords, numclusters, EMB_SIZE, cluster_sizes);
			update_centroids(words, centroids, wordcent, numwords, numclusters, EMB_SIZE, cluster_sizes);
		}  


		// B. Sailkatzearen "kalitatea" -- "Calidad" del cluster
		// =====================================================
		printf("Kalitatea -- Calidad\n");   
		for (i=0; i<numclusters; i++)  members[i].number = 0;

		// cluster bakoitzeko hitzak (osagaiak) eta kopurua -- palabras de cada clusters y cuantas son
		for (i=0; i<numwords; i++)  {
			cluster = wordcent[i];
			zenb = members[cluster].number;
			members[cluster].elements[zenb] = i;	// clusterreko hitza -- palabra del cluster
			members[cluster].number ++; 
		}

		/****************************************************************************************
		  OSATZEKO - PARA COMPLETAR
		  cvi = validation (OSATZEKO - PARA COMPLETAR);
		  if (cvi appropriate) end classification;
		  else  continue classification;	
		 ****************************************************************************************/
		cvi = validation(words, members, centroids, numclusters);
		printf("cvi = %f eta numclusters = %d\n", cvi, numclusters);

		dif = cvi - cvi_zaharra;
		if (dif < DELTA && numclusters > K) {
			end_classif = 1;
		} else {
			cvi_zaharra = cvi;
			numclusters += 10;
		}
	}

	clock_gettime (CLOCK_REALTIME, &t1);
	/******************************************************************/

	for (i=0; i<numclusters; i++)
		printf ("%d. cluster, %d words \n", i, cluster_sizes[i]);

	tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
	printf("\n Tej. (paraleloan) = %1.3f ms\n\n", tej*1000);

	// Idatzi clusterrak fitxategietan -- Escribir los clusters en el fichero
	f3 = fopen (argv[3], "w");
	if (f3 == NULL) {
		printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[3]);
		exit (-1);
	}

	for (i=0; i<numwords; i++)
		fprintf (f3, "%s \t\t -> %d cluster\n", hiztegia[i], wordcent[i]);
	printf ("clusters written\n");

	fclose (f1);
	fclose (f2);
	fclose (f3);

	free(words);
	for (i=0; i<numwords;i++) free (hiztegia[i]);
	free(hiztegia); 
	free(cluster_sizes);
	free(centroids);
	return 0;
}

