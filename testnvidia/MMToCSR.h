#pragma once
#include <stdio.h>
#include <fstream>
#include <string>
#include "progress_bar.h"


void assemble_csr_general_matrix(std::ifstream& fin, int** row_ptr, int** col_ind, int* no_vertices, int* no_edges, bool withData)
{
	int M, N, L;
	fin.ignore(2048, '\n');
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	fin >> M >> N >> L;
	int* row = new int[L];
	int* col = new int[L];
	double data;
	int* offset = (int*)malloc(sizeof(int)*(M + 1));
	int* added = (int*)malloc(sizeof(int)*(M));
	int* col_in_order = (int*)malloc(sizeof(int)*(L));

	for (int i = 0; i < M; i++) {
		added[i] = 0;
		offset[i] = 0;
	}
	offset[M] = 0;
	if (withData)
	{
		for (int i = 0; i < L; i++) {
			fin >> row[i] >> col[i] >> data;
			offset[row[i]]++;
			if (i % 10000 == 0)
				progressBar(i, L);
		}
	}
	else
	{
		for (int i = 0; i < L; i++) {
			fin >> row[i] >> col[i];
			offset[row[i]]++;
			if (i % 10000 == 0)
				progressBar(i, L);
		}
	}
	progressBar(100, 100);
	printf("\nFile read\n");
	fin.close();
	for (int i = 0; i < M; i++) {
		offset[i + 1] += offset[i];
	}
	for (int i = 0; i < L; i++) {
		col_in_order[offset[row[i] - 1] + added[row[i] - 1]] = col[i] - 1;
		added[row[i] - 1]++;
	}
	*col_ind = col_in_order;
	*row_ptr = offset;
	*no_vertices = M;
	*no_edges = L;
	free(added);
	free(col);
	free(row);
	printf("Finished loading data\n");
}
void assemble_csr_symmetric_matrix(std::ifstream& fin, int** row_ptr, int** col_ind, int* no_vertices, int* no_edges, bool withData)
{
	int M, N, L;
	fin.ignore(2048, '\n');
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	fin >> M >> N >> L;
	int* row = new int[L*2];
	int* col = new int[L*2];
	int* offset = (int*)malloc(sizeof(int)*(M + 1));
	int* added = (int*)malloc(sizeof(int)*(M));
	int* col_in_order = (int*)malloc(sizeof(int)*(L*2));
	double data;

	for (int i = 0; i < M; i++) {
		added[i] = 0;
		offset[i] = 0;
	}
	offset[M] = 0;
	if (withData)
	{
		for (int i = 0; i < L; i++) {
			fin >> row[i] >> col[i] >> data;
			row[L + i] = col[i];
			col[L + i] = row[i];
			offset[row[i]]++;
			offset[col[i]]++;
			if (i % 10000 == 0)
				progressBar(i, L);
		}
	}
	else
	{
		for (int i = 0; i < L; i++) {
			fin >> row[i] >> col[i];
			row[L + i] = col[i];
			col[L + i] = row[i];
			offset[row[i]]++;
			offset[col[i]]++;
			if (i % 10000 == 0)
				progressBar(i, L);
		}
	}
	progressBar(99, 100);
	printf("\nFile read\n");
	fin.close();
	for (int i = 0; i < M; i++) {
		offset[i + 1] += offset[i];
	}
	for (int i = 0; i < L*2; i++) {
		col_in_order[offset[row[i] - 1] + added[row[i] - 1]] = col[i] - 1;
		added[row[i] - 1]++;
	}
	*col_ind = col_in_order;
	*row_ptr = offset;
	*no_vertices = M;
	*no_edges = L*2;
	free(added);
	free(col);
	free(row);
	printf("Finished loading data\n");
}

int assemble_csr_matrix(std::string filePath, int** row_ptr, int** col_ind, int* no_vertices, int* no_edges)
{
	printf("Loading %s\n", filePath.substr(0, filePath.length()-4).c_str());
	std::ifstream fileStream(filePath);
	std::string skip;
	std::string patternOrReal;
	std::string generalOrSymmetric;
	fileStream >> skip >> skip >> skip >> patternOrReal >> generalOrSymmetric;
	if (generalOrSymmetric == "general")
	{
		assemble_csr_general_matrix(fileStream, row_ptr, col_ind, no_vertices, no_edges, patternOrReal == "real");
		return 0;
	}
	else if (generalOrSymmetric == "symmetric")
	{
		assemble_csr_symmetric_matrix(fileStream, row_ptr, col_ind, no_vertices, no_edges, patternOrReal == "real");
		return 0;
	}
	else
	{
		printf("Error - invalid file\n");
		return -1;
	}
}