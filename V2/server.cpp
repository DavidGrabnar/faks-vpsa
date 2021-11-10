/*H********************************************************************************
* Ime datoteke: server.cpp
*
* Opis:
*		Enostaven stre�nik, ki zmore sprejeti le enega klienta naenkrat.
*		Stre�nik sprejme klientove podatke in jih v nespremenjeni obliki po�lje
*		nazaj klientu - odmev.
*
*H*/

//Vklju�imo ustrezna zaglavja
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pthread.h>
/*
Definiramo vrata (port) na katerem bo stre�nik poslu�al
in velikost medponilnika za sprejemanje in po�iljanje podatkov
*/
#define PORT 1053
#define BUFFER_SIZE 256
#define DEFAULT_MAX_CONN_COUNT 1;

void* handleClient(void* arg);

pthread_mutex_t mutexCurrConnCount;
int currConnCount = 0;

int main(int argc, char **argv)
{
	int maxConnCount = DEFAULT_MAX_CONN_COUNT;
	if (argc > 1) {
		sscanf(argv[1], "%d", &maxConnCount);
	}

	//Spremenjlivka za preverjane izhodnega statusa funkcij
	int iResult;

	/*
	Ustvarimo nov vti�, ki bo poslu�al
	in sprejemal nove kliente preko TCP/IP protokola
	*/
	int listener = socket(AF_INET, SOCK_STREAM, 0);
	if (listener == -1)
	{
		printf("Error creating socket\n");
		return 1;
	}

	//Nastavimo vrata in mre�ni naslov vti�a
	sockaddr_in listenerConf;
	listenerConf.sin_port = htons(PORT);
	listenerConf.sin_family = AF_INET;
	listenerConf.sin_addr.s_addr = INADDR_ANY;

	//Vti� pove�emo z ustreznimi vrati
	iResult = bind(listener, (sockaddr *)&listenerConf, sizeof(listenerConf));
	if (iResult == -1)
	{
		printf("Bind failed\n");
		close(listener);
		return 1;
	}

	//Za�nemo poslu�ati
	if (listen(listener, 5) == -1)
	{
		printf("Listen failed\n");
		close(listener);
		return 1;
	}

	//Definiramo nov vti� in medpomnilik
	int clientSock;

    pthread_t threadId;

	/*
	V zanki sprejemamo nove povezave
	in jih stre�emo (najve� eno naenkrat)
	*/
	while (1)
	{

		//Sprejmi povezavo in ustvari nov vti�
		clientSock = accept(listener, NULL, NULL);
		if (clientSock == -1)
		{
			printf("Accept failed\n");
			close(listener);
			return 1;
		}

		pthread_mutex_lock(&mutexCurrConnCount);
		int currCount = currConnCount;
		pthread_mutex_unlock(&mutexCurrConnCount);
		
		if (currCount >= maxConnCount) {
			printf("Max connection count reached. Closing new socket\n");
			close(clientSock);
			continue;
		}
		
		int err = pthread_create(&threadId, NULL, handleClient, &clientSock);
		if (err) {
			printf("Failed to create thread: %d", err);
			return err;
		}

		err = pthread_detach(threadId);
		if (err) {
			printf("Failed to detach thread: %d", err);
			return err;
		}
	}

	//Po�istimo vse vti�e
	close(listener);

	return 0;
}

void* handleClient(void* arg)
{
	pthread_mutex_lock(&mutexCurrConnCount);
	currConnCount++;
	pthread_mutex_unlock(&mutexCurrConnCount);

	int clientSock =  *((int *) arg);
	char buff[BUFFER_SIZE];
	int iResult;
	do
	{
		//Sprejmi podatke
		iResult = recv(clientSock, buff, BUFFER_SIZE, 0);
		if (iResult > 0)
		{
			printf("Bytes received: %d\n", iResult);

			//Vrni prejete podatke po�iljatelju
			iResult = send(clientSock, buff, iResult, 0);
			if (iResult == -1)
			{
				printf("send failed!\n");
				close(clientSock);
				break;
			}
			printf("Bytes sent: %d\n", iResult);
		}
		else if (iResult == 0)
			printf("Connection closing...\n");
		else
		{
			printf("recv failed!\n");
			close(clientSock);
			break;
		}

	} while (iResult > 0);

	close(clientSock);
	
	pthread_mutex_lock(&mutexCurrConnCount);
	currConnCount--;
	pthread_mutex_unlock(&mutexCurrConnCount);

	int ret = 0;
	pthread_exit(&ret);
	return nullptr;
}