// ========================================================================= //
//                                                                           //
// Filename: CudaMemoryInfo.h                                                       
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

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <utility>

void postCudaMalloc_Info(bool success, std::size_t size);

void printMemInfo();
void printMemInfo2();

#ifdef WITH_DEVELOPER_INFORMATION
class CudaMemoryInfo
{
public:
	// access
	static CudaMemoryInfo& the()
	{
		static CudaMemoryInfo theInstance;
		return theInstance;
	}
	void restartTracking();
	std::size_t getDeltaGPURAMUsage();
	void trackCurrentGPURAMUsage();

	void startNewUsageInterval(std::string name = "noName");

	void endCurrentUsageInterval();

	void startNewSubInterval();
	void endSubInterval();


	std::size_t getMaximalGPURAMUsage() { return maximalGPURAMUsage; }
	void printGlobalStats();
	void printIntervalStats();
	std::size_t getStartingGPURAMUsage() {
		return startingGPURAMUsage;
	}
private:
	CudaMemoryInfo();
	void updateAllActiveRecords(std::size_t cur);
	void updateRecordPeak(std::size_t cur, size_t recordID);
	void printRecord(size_t currentRecordID, int currentLevel);

	std::size_t maximalGPURAMUsage, startingGPURAMUsage;
	struct memConsumptionRecord
	{
		std::string recordName;
		size_t memIntervalStart;
		size_t memIntervalPeak;
		size_t memIntervalEnd;
	};
	std::vector<memConsumptionRecord> memConsumptionRecords;

	std::vector<int> stackSubInterval; // each index in this "stack" is a pointer to the record in memConsumptionRecord and represents the record one level up. If e.g. stackSubInterval contains 2 and 4, then when processing interval 5, the intervals 2 and 4 must also be updated
	
	bool trackPeakForCurrentRecord = false; // this is false at the beginning and when a user calle endCurrentUsageInterval
};
#endif
