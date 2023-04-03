// ========================================================================= //
//                                                                           //
// Filename: CudaMemoryInfo.cpp
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2012 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 25.01. Tim Grasser
//                                                                           //
// ========================================================================= //

#include "CudaMemoryInfo.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip> // Used because of std::setprecision()
#include <sstream>
#include <iostream>
#include <cassert>


void postCudaMalloc_Info(bool success, std::size_t size)
{
#ifdef WITH_DEVELOPER_INFORMATION
	CudaMemoryInfo::the().trackCurrentGPURAMUsage();
#endif
	if(!success)
	{
		std::size_t free = 0, total = 0;
		cudaMemGetInfo(&free, &total);
		std::stringstream sStream;
		sStream << std::setprecision(3) << std::fixed << "Error: Unable to allocate GPU array with " << static_cast<float>(size) / 1024.0f / 1024.0f
			<< " MiB! Only " << static_cast<float>(free) / 1024.0f / 1024.0f << "MiB out of " << static_cast<float>(total) / 1024.0f / 1024.0f
			<< " MiB GPU memory remaining!\n";
#ifdef WITH_DEVELOPER_INFORMATION
		float startRAM = static_cast<float>(CudaMemoryInfo::the().getStartingGPURAMUsage());
		std::cout << "GPU RAM was initial " << startRAM / 1024.0f / 1024.0f;
#endif
		std::cout << "\n";
		std::cerr << sStream.str();
		throw std::bad_alloc();
	}
}

void printMemInfo()
{
#ifdef WITH_DEVELOPER_INFORMATION
	std::size_t free = 0, total = 0;
	cudaMemGetInfo(&free, &total);
	double BtoGB = 1.0 / (1ull << 30);
	std::printf("%.2fGB of %.2f GB GPU memory available, %.2fGB in use\n", BtoGB * free, BtoGB * total, BtoGB * (total - free));
#endif
}

void printMemInfo2()
{
#ifdef WITH_DEVELOPER_INFORMATION
	std::size_t free = 0, total = 0;
	cudaMemGetInfo(&free, &total);
	double BtoMB = 1.0 / (1ull << 20);
	std::printf("%.2fMB of %.2f MB GPU memory available, %.2fMB in use\n", BtoMB * free, BtoMB * total, BtoMB * (total - free));
	double BtoKB = 1.0 / (1ull << 10);
	std::printf("%.2fKB of %.2fKB GPU memory available, %.2fKB in use\n", BtoKB * free, BtoKB * total, BtoKB * (total - free));
#endif
}


#ifdef WITH_DEVELOPER_INFORMATION

std::size_t getCurrentGPURAMUsage()
{
	std::size_t free = 0, total = 0;
	cudaMemGetInfo(&free, &total);
	return total - free;
}

CudaMemoryInfo::CudaMemoryInfo()
	:maximalGPURAMUsage(0)
	,startingGPURAMUsage(0)
{}

std::size_t CudaMemoryInfo::getDeltaGPURAMUsage()
{
	auto cur = getCurrentGPURAMUsage();
	return cur - this->startingGPURAMUsage;
}

void CudaMemoryInfo::restartTracking()
{
	startingGPURAMUsage = getCurrentGPURAMUsage();
	memConsumptionRecords.clear();
	startNewUsageInterval("baseline");
}

void CudaMemoryInfo::startNewUsageInterval(std::string name)
{
	auto cur = getDeltaGPURAMUsage();
	if(!memConsumptionRecords.empty() && trackPeakForCurrentRecord)
	{
		updateAllActiveRecords(cur);
		
		printRecord(memConsumptionRecords.size() - 1, static_cast<int>(stackSubInterval.size()));
	}
	memConsumptionRecords.push_back({name, cur, cur, cur});

	trackCurrentGPURAMUsage();
	trackPeakForCurrentRecord = true;
}

void CudaMemoryInfo::updateAllActiveRecords(std::size_t cur)
{
	for(auto recordID : stackSubInterval)
		updateRecordPeak(cur, recordID);
	updateRecordPeak(cur, memConsumptionRecords.size() - 1);
	memConsumptionRecords.back().memIntervalEnd = cur;
}

void CudaMemoryInfo::updateRecordPeak(std::size_t cur, size_t recordID)
{
	auto && currentRecord = memConsumptionRecords[recordID];
	auto && peak = currentRecord.memIntervalPeak;
	peak = std::max(peak, cur);
}

void CudaMemoryInfo::endCurrentUsageInterval()
{
	trackPeakForCurrentRecord = false;
	auto cur = getDeltaGPURAMUsage();
	if(!memConsumptionRecords.empty()) 
	{
		updateAllActiveRecords(cur);
	}
}

void CudaMemoryInfo::startNewSubInterval()
{
	trackPeakForCurrentRecord = false; // as we are in a new sub interval, there is a new start
	auto currentRecordID = static_cast<int>(memConsumptionRecords.size()) - 1;
	stackSubInterval.push_back(currentRecordID);
}

void CudaMemoryInfo::endSubInterval()
{
	if(stackSubInterval.empty())
	{
		std::fprintf(stderr, "Error: Ending subinterval, but no subinterval started\n!");
		assert(0);
		std::abort();
	}

	endCurrentUsageInterval();

	auto currentRecordID = memConsumptionRecords.size() - 1;
	auto currentLevel = static_cast<int>(stackSubInterval.size());

	printRecord(currentRecordID, currentLevel);


	auto parentRecordID = stackSubInterval.back();
	stackSubInterval.pop_back();

	auto cur = getDeltaGPURAMUsage();
	updateAllActiveRecords(cur);

	auto && parentRecord = memConsumptionRecords[parentRecordID];
	parentRecord.memIntervalEnd = cur;

	std::printf("X");
	printRecord(parentRecordID, static_cast<int>(stackSubInterval.size()));
}

void CudaMemoryInfo::printRecord(size_t recordID, int currentLevel)
{
	auto && record = memConsumptionRecords[recordID];
	std::printf("MEML%d %s, (begin, end, add peak, delta): %.4f %.4f %.4f %.4f\n",
		currentLevel,
		record.recordName.c_str(),
		record.memIntervalStart / 1024.0f / 1024.0f,
		record.memIntervalEnd / 1024.0f / 1024.0f,
		((float)record.memIntervalPeak - (float)record.memIntervalEnd) / 1024.0f / 1024.0f, 
		((float)record.memIntervalEnd - (float)record.memIntervalStart) / 1024.0f / 1024.0f
		);
}

void CudaMemoryInfo::trackCurrentGPURAMUsage()
{
	auto cur = getDeltaGPURAMUsage();
	maximalGPURAMUsage = std::max(maximalGPURAMUsage, cur);
	if(!memConsumptionRecords.empty())
		updateAllActiveRecords(cur);
	//if(!memConsumptionRecords.empty())
	//{
	//	auto && peak = memConsumptionRecords.back().memIntervalPeak;
	//	peak = std::max(peak, cur);
	//}
}

void CudaMemoryInfo::printGlobalStats()
{
	float startRAM = static_cast<float>(startingGPURAMUsage), maxRAM = static_cast<float>(maximalGPURAMUsage);
	std::cout << "GPU RAM was initial " << startRAM / 1024.0f / 1024.0f << " MiB, max "
		<< maxRAM / 1024.0f / 1024.0f << " MiB, so effective used was "
		<< (maxRAM - startRAM) / 1024.0f / 1024.0f << " MiB.\n";
}

void CudaMemoryInfo::printIntervalStats()
{
	if(!memConsumptionRecords.empty())
	{
		std::cout << "Intermediate usage of GPU RAM was: \n";
		for(auto && p : memConsumptionRecords)
		{
			std::cout << p.recordName << "\n";
			//std::cout << "    RAM usage at start: " << static_cast<float>(p.memIntervalStart) / 1024.0f / 1024.0f << " MiB\n";
			std::cout << "    Additional RAM usage during Interval : " << (static_cast<float>(p.memIntervalPeak) - static_cast<float>(p.memIntervalStart)) / 1024.0f / 1024.0f << " MiB\n";
			//std::cout << "    RAM usage at end: " << static_cast<float>(p.memIntervalEnd) / 1024.0f / 1024.0f << " MiB\n";
			std::cout << "    Delta RAM start to end interval : " << (static_cast<float>(p.memIntervalEnd) - static_cast<float>(p.memIntervalStart)) / 1024.0f / 1024.0f << " MiB\n";
		}
	}
}
#endif
