#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

//DEKLARACAJE FUNKCJI
void initMatrix();
void fillMatrix();
void showMatrix();
void initRegisterState();
void initUsedRegisterValues();
void startCalculations();
void calculateCycle();
void addToUsedRegisterValues();
void calculateRegisterState();
bool calculateCell(bool* matrixRow);
bool isAlreadyUsedState();
bool isEveryRegisterStateUsed();
int parseRegisterState();
void convertToRegisterState(int decimalNumber);

using namespace std;

bool** conversionMatrix;
bool* currentRegisterState;
bool* usedRegisterValues;
int flipFlopCount = 0;
bool maximalCycle = false; //2n-1

long time_start, time_stop;

//FLAGI

bool isEveryRegisterStateUsed()
{
	bool result = true;

	int n = pow(2, flipFlopCount) - 1;

	for (int i = 1; i <= n; i++)
		if (usedRegisterValues[i] == false)
		{
			result = false;
			break;
		}

	return result;
}

bool isAlreadyUsedState()
{
	return usedRegisterValues[parseRegisterState()];
}

//INIT

void initMatrix()
{
	cout << "Podaj wielkoœæ macierzy: ";
	cin >> flipFlopCount;
	cout << endl << endl;

	cudaMallocManaged(&conversionMatrix, pow(flipFlopCount, 2) * sizeof(bool));

	//INITIALIZING MATRIX
	conversionMatrix = new bool *[flipFlopCount];
	for (int i = 0; i < flipFlopCount; i++)
	{
		conversionMatrix[i] = new bool[flipFlopCount];
	}
	//INITIALIZING MATRIX - END
}

void initUsedRegisterValues()
{
	int n = pow(2, flipFlopCount); // na 3 bitach jest 8 stanów - nie 7
	usedRegisterValues = new bool[n]{false};
	usedRegisterValues[0] = 1;
}

void initRegisterState()
{
	cudaMallocManaged(&currentRegisterState, flipFlopCount * sizeof(bool));
	currentRegisterState = new bool[flipFlopCount];
	//currentRegisterState[0] = 1;
	for (int i = 0; i < flipFlopCount; i++)
	{
		currentRegisterState[i] = 0;
	}
}

void printRegisterState()
{
	for (int i = flipFlopCount-1; i >= 0; i--)
	{
		cout << currentRegisterState[i] ;
	}
	cout<< endl;
}

//MATRIX

void fillMatrix()
{
	for (int i = 0; i < flipFlopCount; i++)
	{
		cout << "Podaj " << i + 1 << " wiersz: ";

		for (int j = 0; j < flipFlopCount; j++)
		{
			cin >> conversionMatrix[i][j];
		}
		cout << endl;
	}
}

void showMatrix()
{
	cout << endl;

	for (int i = 0; i < flipFlopCount; i++)
	{
		for (int j = 0; j < flipFlopCount; j++)
		{
			cout << conversionMatrix[i][j] << " ";
		}

		cout << endl;
	}
}

__global__ void calculateCellParallel(bool* currentRS, bool* nextRS, bool** matrix, int n)
{
	int result = 0;
	
	for (int i = 0; i < n; i++)
	{
		result = result + (matrix[threadIdx.x][i] * currentRS[i]);
		
	}
	
	 nextRS[threadIdx.x] = (result % 2);
}

bool calculateCell(bool* matrixRow)
{
	int result = 0;

	for (int i = 0; i < flipFlopCount; i++)
	{
		result = result + (matrixRow[i] * currentRegisterState[i]);
	}

	return (result % 2);
}

void calculateRegisterState()
{
	bool* nextRegister = new bool[flipFlopCount];

	for (int i = 0; i < flipFlopCount; i++)
	{
		nextRegister[i] = calculateCell(conversionMatrix[i]);
	}

	currentRegisterState = nextRegister;
}

void calculateRegisterStateParallel()
{
	cout << "START ";

	bool* nextRegisterState;

	cudaMallocManaged(&nextRegisterState, flipFlopCount * sizeof(bool));
	// cudaMallocManaged(&conversionMatrix, pow(flipFlopCount, 2) * sizeof(bool));
	// cudaMallocManaged(&currentRegisterState, flipFlopCount * sizeof(bool));

	nextRegisterState = new bool[flipFlopCount]{false};


	cout << "GPU";
	calculateCellParallel <<<1, flipFlopCount>>>(currentRegisterState, nextRegisterState, conversionMatrix,
	                                             flipFlopCount);


	cudaDeviceSynchronize();
	cout << " STOP ";

	*currentRegisterState = nextRegisterState;

	cudaFree(nextRegisterState);
	printRegisterState();
}

//LOGIKA


void calculateCycle()
{
	//przemno¿yæ macierz przez rejestr - mamy rejestr wynikowy
	int statesNumber = 0;
	do
	{
		statesNumber++;
		addToUsedRegisterValues();
		calculateRegisterStateParallel();
	}
	while ((parseRegisterState() != 0) && !isAlreadyUsedState());
	//sprawdziæ czy rejestr przedstawia zero jeœli jest lub czy jest w u¿ytych (parsowane na liczbe) - jesli tak koniec
	cout << "liczba stanów w cyklu: " << statesNumber << endl;
	if (statesNumber == pow(2, flipFlopCount) - 1)
	{
		maximalCycle = true;
	}
}

int parseRegisterState()
{
	//metoda parsuj¹ca
	int result = 0;
	for (int i = 0; i < flipFlopCount; i++)
		if (currentRegisterState[i])
			result += pow(2, i);

	return result;
}

void addToUsedRegisterValues()
{
	usedRegisterValues[parseRegisterState()] = true;
}

void findFirstAvailableState()
{
	int n = pow(flipFlopCount, 2) - 1;

	for (int i = 1; i <= n; i++)
	{
		if (!usedRegisterValues[i])
		{
			convertToRegisterState(i);
			break;
		}
	}
}

void convertToRegisterState(int decimalNumber)
{
	int i = 0;
	do
	{
		currentRegisterState[i] = decimalNumber % 2;
		decimalNumber /= 2;
		i++;
	}
	while (decimalNumber > 0);
}

void startCalculations()
{
	time_start = std::chrono::system_clock::now().time_since_epoch().count();


	//Pocz¹tek wszystkich obliczeñ
	int cyclesNumber = 0;
	//pocz¹tek liczenia jednego cyklu
	do
	{
		//obecny rejestr idzie do u¿ytych rejestrów
		findFirstAvailableState();
		calculateCycle();
		//koniec pêtli
		cyclesNumber++;
	}
	while (!isEveryRegisterStateUsed());

	cout << endl << "Finalna liczba cykli: " << cyclesNumber << endl;

	if (maximalCycle)
	{
		cout << endl << "generuje cykl maksymalny" << endl;;
	}
	time_stop = (std::chrono::system_clock::now().time_since_epoch().count());
	cout << "Algorytm trwa³: " << time_stop - time_start << "ms" << endl;
}


//MAIN

int main()
{
	
	initMatrix();
	fillMatrix();
	showMatrix();

	initRegisterState();
	initUsedRegisterValues();

	startCalculations();

	getchar();
	return 0;
}
