/*
Castillo Guzmán Luis Eduardo / 5BM1
Primer Evaluación Práctica Visión Artificial
*/
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

//Imprimir en excel:
void print_excel(Mat img, char path[]) {
	FILE* i_orig = fopen(path, "w");

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {
			fprintf(i_orig, "%f \t", (img.at<float>(i, j)));
		}

		fprintf(i_orig, "\n");
	}
	fclose(i_orig);
}

Mat floatToUchar(Mat fmat) {
	/*Funcion que cambia de tipo de dato float a unsigned char una matriz de openCV
	Se usa al final de los procesos*/

	Mat umat;
	fmat.convertTo(umat, CV_8U);
	return umat;
}

Mat ucharToFloat(Mat umat) {
	/*Funcion que cambia de tipo de dato uchar a float una matriz de openCV
	Se usa principalmente para evitar realizar multiples cast durante los procesos
	al final de cada proceso normalmente se usa floatToUchar para representar la matriz 
	en tipo de dato uchar*/

	Mat fmat;
	umat.convertTo(fmat, CV_32F);
	return fmat;
}

Mat aumentar_imagen(Mat img, int kernel_size) {
	/*Funcion que añade bordes a una imagen*/
	img = ucharToFloat(img); //Trabajamos con la imagen en float

	int mc = (kernel_size - 1) / 2; //Este dato nos sirve para saber cuantos pixeles añadiremos a los bordes

	int src_rows = img.rows;
	int src_cols = img.cols;

	//Creamos la matriz aumentada, se añaden 2 en cada extremo de la imagen, por eso se multiplica por dos
	Mat imresized = Mat::zeros(src_rows + mc * 2, src_cols + mc * 2, CV_32F); 

	for (int i = 0; i < src_rows; i++)
		for (int j = 0; j < src_cols; j++)
			imresized.at<float>(i + mc, j + mc) = img.at<float>(i, j); //Se asigna

	return floatToUchar(imresized); //Regresamos la imagen aumentada en uchar
}

Mat gaussian_kernel(int n, float sigma) {
	int mc = (n - 1) / 2; //Sirve para obtener las coordenadas x,y en conjunto con los indices i,j

	Mat gk = Mat::zeros(n, n, CV_32F); //Gauss Kernel

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			int x = j - mc;
			int y = -i + mc;

			//Calculamos y asignamos:
			gk.at<float>(i, j) = float(exp((-1 * (pow(x, 2) + pow(y, 2))) / (2 * pow(sigma, 2))) / (2 * 3.1416 * pow(sigma, 2)));
		}

	return gk;
}

Mat aplicar_filtro(Mat img_resized, Mat kernel, bool normalizar = false, bool neg=false) {
	img_resized = ucharToFloat(img_resized);

	int mc = (kernel.rows - 1) / 2; //Este número nos sirve para recorrer el kernel
	int rows = img_resized.rows - mc * 2; //Las filas de la imagen original
	int cols = img_resized.cols - mc * 2; //Las columnas de la imagen original
	Mat img_filtered = Mat(rows, cols, CV_32F); //Creamos la matriz de la imagen filtrada

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			float suma = 0.0f; //Aquí se guardará la suma de los valores que se vayan multiplicando del kernel con la region de la imagen correspondiente
			for (int ik = 0; ik < kernel.rows; ik++) //Recorre al kernel
				for (int jk = 0; jk < kernel.cols; jk++) {
					//Estas coordenadas y,x se suman a i,j para obtener los valores correspondientes de la imagen y del kernel
					//para multiplicarlos entre si y añadir el resultado a la suma.
					int y = ik - mc; 
					int x = jk - mc; 

					float valor_pixel = img_resized.at<float>(i + y + mc, j + x + mc);
					float valor_kernel = kernel.at<float>(ik, jk);
					suma = suma + valor_pixel * valor_kernel;
				}
			if (suma > 255)
				suma = 255;
			if (neg) { 
				img_filtered.at<float>(i, j) = suma; //Se asignan valores negativos (derivadas del sobel)
			}
			else {
				img_filtered.at<float>(i, j) = abs(suma);
			}
			
		}

	if (neg) {
		return img_filtered; //Se regresa la imagen filtrada DE TIPO FLOAT
	}

	/*En caso de querer normalizar (filtro gaussiano) se tienen que dividir todos los valores de la imagen
	entre la suma total de los valores del kernel correspondiente*/
	if (normalizar) {
		float total_k = 0;
		for (int i = 0; i < kernel.rows; i++)
			for (int j = 0; j < kernel.cols; j++)
				total_k = total_k + kernel.at<float>(i, j);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				img_filtered.at<float>(i, j) = (img_filtered.at<float>(i, j)) / total_k;
	}

	return floatToUchar(img_filtered); //Se regresa la imagen filtrada en uchar
}

Mat filtro_gaussiano(Mat img, int size, float sigma) {
	Mat img_resized = aumentar_imagen(img, size);  //Se añaden bordes
	Mat gau_kernel = gaussian_kernel(size, sigma); //Se obtiene el kernel gauss
	Mat smoothed = aplicar_filtro(img_resized, gau_kernel, true); //Se aplica el kernel (con normalizacion)

	cout << "Gauss Kernel: "<<size<<"x"<<size <<" , Sigma: "<<sigma << endl << gau_kernel << endl;

	imshow("Imagen escala grises aumentada", img_resized);
	cout << "Dimensiones de Imagen escala grises aumentada: " << img_resized.rows << " filas, " << img_resized.cols << " columnas." << endl;

	return smoothed;
}

Mat G(Mat sobel_x, Mat sobel_y) {
	/*Este funcion obtiene |G|*/
	Mat sobel = Mat(sobel_x.rows, sobel_x.cols, CV_32F);

	//Se usa la formula |G| = (Gx^2 + Gy^2)^(1/2)
	for (int i = 0; i < sobel_x.rows; i++)
		for (int j = 0; j < sobel_x.cols; j++)
			sobel.at<float>(i, j) = float(sqrt(pow(sobel_x.at<float>(i, j), 2) + pow(sobel_y.at<float>(i, j), 2)));
	return floatToUchar(sobel);
}

Mat theta(Mat sobel_x, Mat sobel_y) {
	/*En este caso sobel_x y sobel_y tienen valores NEGATIVOS, esto debido a que los necesitamos ya que 
	de no tenerlos el resultado de la operacion arcotangente que se realiza en este proceso siempre resultará 
	en valores menores e iguales a 90 grados, por lo tanto para arreglar esto se tiene que sumar 180 al resultado
	en caso de ser negativo. De esta manera tendremos angulos adecuados para el siguiente proceso (Non_Maximum_Supression)*/

	Mat angulo = Mat(sobel_x.rows, sobel_x.cols, CV_32F);

	for (int i = 0; i < sobel_x.rows; i++)
		for (int j = 0; j < sobel_x.cols; j++) {
			float pixelGx = sobel_x.at<float>(i, j);
			float pixelGy = sobel_y.at<float>(i, j);
			float arcotangente = 0.0f;

			if ((int)pixelGx != 0) { //No se indetermina la division de Gy/Gx
				arcotangente = float(atan(pixelGy / pixelGx) * (180 / 3.1416));
			} 

			if (arcotangente < 0) {
				arcotangente = arcotangente+180;
			}
			angulo.at<float>(i, j) =  arcotangente;
		}
	return floatToUchar(angulo);
}

Mat Non_Maximum_Supression(Mat G, Mat ang) {
	/*Este metodo sirve para adelgazar los bordes encontrados en |G| usando esta matriz y tambien la matriz
	de theta, para ello evaluamos el valor de cada pixel en la matriz theta y dependiendo de este angulo
	tomaremos en cuenta una vecindad específica (diagonal, horizontal o vertical). Dentro de esta vecindad
	tendremos dos valores de pixel que compararemos con el que esta siendo procesado en la matriz |G|, si
	alguno de estos es mayor a este valor, entonces reemplazaremos este valor con 0, de lo contrario el valor
	sera el mismo.*/
	G = ucharToFloat(G);
	ang = ucharToFloat(ang);

	int M = G.rows;
	int N = G.cols;

	Mat NMS = Mat::zeros(M, N, CV_32F);

	for(int i=2; i<M-1; i++)
		for (int j = 1; j < N-1; j++) {
			float p1, p2;

			//Horizontal:
			if ((0 <= ang.at<float>(i, j) < 22.5) || (157.5 <= ang.at<float>(i, j) <= 180)) {
				p1 = G.at<float>(i, j + 1);
				p2 = G.at<float>(i, j - 1);
			}
			//Diagonal (45 grados):
			else if (22.5<= ang.at<float>(i, j)<67.5) {
				p1 = G.at<float>(i + 1, j-1);
				p2 = G.at<float>(i - 1, j+1);
			}
			//Vertical:
			else if (67.5 <= ang.at<float>(i, j) < 112.5) {
				p1 = G.at<float>(i + 1, j);
				p2 = G.at<float>(i - 1, j);
			}
			//Diagonal (135 grados)
			else if (112.5 <= ang.at<float>(i, j) < 157.5) {
				p1 = G.at<float>(i - 1, j - 1);
				p2 = G.at<float>(i + 1, j + 1);
			}

			if ((G.at<float>(i, j) >= p1) && (G.at<float>(i, j) >= p2)) {
				NMS.at<float>(i, j) = G.at<float>(i, j);
			}
			else {
				NMS.at<float>(i, j) = 0;
			}
		}

	return floatToUchar(NMS);
}

Mat threshold(Mat img, float lowTR = 0.25f, float highTR = 0.45f) {
	/*Esta funcion realiza dos umbralizado, uno alto y otro bajo, la idea es que todos los valores que esten por
	debajo del umbral bajo se consideren irrelevantes y por tanto se toman como 0, despues los valores que son mayores
	al umbral bajo pero menores al umbral alto se consideran valores debiles y se les asigna un valor de 25, por último
	para los valores mayores al umbral alto se les asigna 255. En general los parametros lowTR y highTR que controlan 
	los umbrales suelen afectar bastante en el resultado final.*/
	img = ucharToFloat(img);
	double maxValue, minValue;
	minMaxLoc(img, &minValue, &maxValue); //Sacamos el valor maximo de la imagen

	double highThreshold = maxValue * highTR; //Calculamos el umbral alto
	double lowThreshold = highThreshold * lowTR; //Umbral bajo

	int M = img.rows;
	int N = img.cols;

	Mat res = Mat(M, N, CV_32F);

	//Se realiza el umbralizado:
	for(int i=0; i<M; i++)
		for (int j = 0; j < N; j++) {
			if (img.at<float>(i, j) >= highThreshold) {
				res.at<float>(i, j) = 255;
			}
			else if ((img.at<float>(i, j) <= highThreshold) && (img.at<float>(i, j) >= lowThreshold)) {
				res.at<float>(i, j) = 25;
			}
			else {
				res.at<float>(i, j) = 0;
			}
		}

	return floatToUchar(res);
}

Mat hysteresis(Mat img, int weak, int strong) {
	/*La histeresis es el último proceso del operador Canny, consiste en realizar una conexión entre los pixeles
	debiles y fuertes definidos en el proceso de umbralizado, basicamente se procesa cada pixel de la imagen umbralizada
	si este pixel es debil evaluaremos su vecindad, si dentro de esta existe algun pixel fuerte entonces este valor se
	cambiara por un valor del pixel fuerte, de lo contrario sera cero*/
	img = ucharToFloat(img);

	int M = img.rows;
	int N = img.cols;

	for (int i = 1; i < M-1; i++)
		for (int j = 1; j < N-1; j++) {
			if ((int)img.at<float>(i, j) == weak) {
				//Si el pixel es debil, evaluamos su vecindad:
				if (((int)img.at<float>(i - 1, j - 1) == strong) || ((int)img.at<float>(i - 1, j) == strong)
				|| ((int)img.at<float>(i - 1, j + 1) == strong)|| ((int)img.at<float>(i, j - 1) == strong)
				|| ((int)img.at<float>(i, j) == strong)|| ((int)img.at<float>(i, j + 1) == strong)
				|| ((int)img.at<float>(i + 1, j - 1) == strong)|| ((int)img.at<float>(i + 1, j) == strong)
				|| ((int)img.at<float>(i + 1, j + 1) == strong)) {
				img.at<float>(i, j) = (float)strong; //Si hay algun valor fuerte entonces se le asigna uno
				}
				else {
					img.at<float>(i, j) = 0.0f; //Si no entonces sera cero
				}
			}
			
		}
	return floatToUchar(img);
}

Mat Canny_edges(Mat img) {
	Mat img_resized = aumentar_imagen(img, 3);

	float skx_data[9] = { -1,0,1 ,-2,0,2,-1,0,1 };
	Mat skx = Mat(3, 3, CV_32F, skx_data); //Sobel kernel horizontal (Gx) 

	float sky_data[9] = { 1,2,1 ,0,0,0,-1,-2,-1 };
	Mat sky = Mat(3, 3, CV_32F, sky_data); //Sobel kernel vertical (Gy)

	//Se aplican los filtros:
	Mat sobel_x = aplicar_filtro(img_resized, skx, false, true);
	Mat sobel_y = aplicar_filtro(img_resized, sky, false, true);

	Mat sobel = G(sobel_x, sobel_y); //Se obtiene |G|
	Mat ang = theta(sobel_x, sobel_y); //Se obtiene theta = arctan(Gy/Gx)
	
	Mat NMS = Non_Maximum_Supression(sobel, ang);

	Mat img_thresh = threshold(NMS);
	Mat canny = hysteresis(img_thresh, 25, 255); //Weak:25 , strong:255

	imshow("Imagen ecualizada aumentada", img_resized);
	cout << "Dimensiones de Imagen ecualizada aumentada: " << img_resized.rows << " filas, " << img_resized.cols << " columnas." << endl;
	imshow("Sobel Gx", floatToUchar(sobel_x));
	cout << "Dimensiones de Sobel Gx: " << sobel_x.rows << " filas, " << sobel_x.cols << " columnas." << endl;
	imshow("Sobel Gy", floatToUchar(sobel_y));
	cout << "Dimensiones de Sobel Gy: " << sobel_y.rows << " filas, " << sobel_y.cols << " columnas." << endl;
	imshow("|G|", sobel);
	cout << "Dimensiones de |G|: " << sobel.rows << " filas, " << sobel.cols << " columnas." << endl;
	imshow("theta", ang);
	cout << "Dimensiones de theta: " << ang.rows << " filas, " << ang.cols << " columnas." << endl;
	imshow("Non Maximum Supression", NMS);
	cout << "Dimensiones de Non Maximum Supression: " << NMS.rows << " filas, " << NMS.cols << " columnas." << endl;
	imshow("Thresholding", img_thresh);
	cout << "Dimensiones de Thresholding: " << img_thresh.rows << " filas, " << img_thresh.cols << " columnas." << endl;

	return canny;
}

int main() {
	int size;
	float sigma;

	cout << "Introduzca el size del kernel (3,5,7,...): ";
	cin >> size;
	cout << "Introduzca sigma: ";
	cin >> sigma;

	Mat img_original = imread("Images/lena.png");
	imshow("Imagen Original", img_original);
	cout << "Dimensiones de Imagen original: " << img_original.rows << " filas, " << img_original.cols << " columnas." << endl;

	Mat img;
	cvtColor(img_original, img, COLOR_BGR2GRAY); //Imagen escala de grises
	imshow("Imagen escala de grises", img);
	cout << "Dimensiones de Imagen escala de grises: " << img.rows << " filas, " << img.cols << " columnas." << endl;

	Mat smoothed = filtro_gaussiano(img, size, sigma); //Imagen suavizada
	imshow("Imagen suavizada", smoothed);
	cout << "Dimensiones de Imagen suavizada: " << smoothed.rows << " filas, " << smoothed.cols << " columnas." << endl;

	Mat imagen_ecualizada;
	equalizeHist(smoothed, imagen_ecualizada); //Ecualización
	imshow("Imagen Ecualizada", imagen_ecualizada);
	cout << "Dimensiones de Imagen ecualizada: " << imagen_ecualizada.rows << " filas, " << imagen_ecualizada.cols << " columnas." << endl;

	Mat canny = Canny_edges(imagen_ecualizada); //Canny
	imshow("Imagen detección de borde Canny", canny);
	cout << "Dimensiones de Imagen deteccion de borde Canny: " << canny.rows << " filas, " << canny.cols << " columnas." << endl;

	waitKey(0);
	return 0;
}