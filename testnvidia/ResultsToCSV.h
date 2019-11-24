#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "defines.h"
#include <chrono>
#include <ctime>

bool CheckCSV(char* fileName, std::string solutions[])
{
	std::ifstream file(fileName);
	std::string delimiter = ";";
	std::string header;
	if (file.good() && std::getline(file, header))
	{
		std::vector<std::string> headerCols;
		size_t pos = 0;
		while ((pos = header.find(delimiter)) != std::string::npos) {
			headerCols.push_back(header.substr(0, pos));
			header.erase(0, pos + delimiter.length());
		}
		int noConstCols = 8;
		for (int i = noConstCols; i < headerCols.size(); i++)
		{
			if (headerCols[i] != solutions[i - noConstCols])
			{
				printf("Invalid header in csv file: found %s where %s was expected\n", headerCols[i].c_str(), solutions[i - noConstCols].c_str());
				exit(0);
			}
		}
		file.close();
		return true;
	}
	return false;
}

void SaveResults(char* fileName, std::string graphFileName, int noVertices, int noEdges, int noTests, float nvgraphResult, float CPUResult, std::string solutions[], float results[], bool errorEncountered[], int noSolutions)
{
	std::fstream file;
	std::string delimiter = ";";
	std::string header;

	if (CheckCSV(fileName, solutions))
	{
		file.open(fileName, std::ios::out | std::ios::app);
	}
	else
	{
		file.open(fileName, std::ios::out | std::ios::trunc);
		file << "time" << delimiter << "graph" << delimiter << "no_vertices" << delimiter;
		file << "no_edges" << delimiter << "no_tests" << delimiter << "block_size" << delimiter;
		file << "nvgraph" << delimiter << "CPU" << delimiter;
		for (size_t i = 0; i < noSolutions; i++)
		{
			file << solutions[i] << delimiter;
		}
		file << std::endl;
	}
	auto time = std::chrono::system_clock::now();
	std::time_t time_t = std::chrono::system_clock::to_time_t(time);
	file << std::to_string(time_t) << delimiter << graphFileName.substr(0, graphFileName.length() - 4).c_str() << delimiter;
	file << noVertices << delimiter << noEdges << delimiter << noTests << delimiter << NO_THREADS << delimiter << nvgraphResult << delimiter;
	file << CPUResult << delimiter;
	for (size_t i = 0; i < noSolutions; i++)
	{
		if (errorEncountered[i])
		{
			file << "ERROR" << delimiter;
		}
		else
		{
			file << results[i] << delimiter;
		}
	}
	file << std::endl;
}