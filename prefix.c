
/* 
 * PARALLEL PREFIX SCAN ALGORITHM IMPLEMENTATION
 * written by: Aniebiet Akpan, 20212007
 * for: Advanced Computational Engineering
 * on: Concurrent Programming using MPI
 * University of Nottingham
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/*
 *Program is designed to run using at least 4 processes
 *with maximum number of input set to 10000
 *
 *to run the program use: mpicc -np [number of processes] ./[Exec Name] [number of input]
*/

int checkInputError(int agc, char *agv[], int worldsize);
int getRank(int columnId, int worldsize);
int getPaddedCount(int, int worldsize);
void pre_scan(int dataCount);

int main(int argc, char *argv[])
{
    int rank, size;
	
    MPI_Init(&argc, &argv);
    
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get world size
	
	if(rank == 0) if(checkInputError(argc, argv, size)) exit(1);     //validate execution command
	
	int dataCount = atoi(argv[1]); //get number of input: N, maximum is set to 10000
	
	pre_scan(dataCount); //call pre_scan to compute scan
    
	MPI_Finalize();
    return 0;
}

void pre_scan(int dataCount){
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	int PROCESSES = size;

	int padded_data_count = getPaddedCount(dataCount, PROCESSES); //do padding and get data count
	int n_inputs = padded_data_count/PROCESSES; //number of inputs to each process
	
	//timing variables, timing is done by reduction, so time from each rank is compiled in global variable
	double upTimeStart, upTimeEnd, upTimeDuration, upTimeGlobal;
	double downTimeStart, downTimeEnd, downTimeDuration, downTimeGlobal;
	double scatterTimeStart, scatterTimeEnd, scatterTimeDuration, scatterTimeGlobal;
	double gatherTimeStart, gatherTimeEnd, gatherTimeDuration, gatherTimeGlobal;
	double serialTimeStart, serialTimeEnd, serialTimeDuration;
	
	//generate random data for supplied datacount
	int *inputData = (int*)malloc(padded_data_count * sizeof(int)); //allocate mem for input data
	int *outputData = (int*)malloc(padded_data_count * sizeof(int)); //allocate mem for output data after down phase is complete
	int *outputDataUp = (int*)malloc(padded_data_count * sizeof(int)); //allocate mem for output data after up phase is complete	
	int *processInput = (int*)malloc(n_inputs * sizeof(int)); //allocate mem for input data for each downphase process
	int *processInputDown = (int*)malloc(n_inputs * sizeof(int)); //allocate mem for input data for each process
	
	//confirm successful mem allocation
	if (!(inputData || outputData || outputDataUp || processInput || processInputDown )){
		fprintf(stderr, "\r\nError allocating memory, check heap space. \r\n");
		return;
	}
	
	//generate input
	int i,j;
	for(i=0; i<dataCount; i++){
		*(inputData + i) = rand() % 10 + 1; //use this line for random data
		//*(inputData + i) = i + 1; //use this line for serial data
	}
	
	//fill the padding with 0s
	for(j=dataCount; j<padded_data_count; j++){
		*(inputData + j) = 0;
	}
	
	//print input data to user
	if(rank == 0){
		printf("\nINPUT DATA\n");
		for(i=0; i<dataCount; i++){
			printf(" %d ", *(inputData+i));
		}
	}
	
	//viewing the grid as a matrix of Dimension N by log N //base 2
	//lets assume each column has an ID given by: [Nth column for that process] * world_size + rank
	//e.g for 9th element say using 8 processes (starting at 0), columnID is 1 * 8 + 1 = 9 
	int n_steps = ceil(log(dataCount)/log(2)); //number of rows i.e steps
	
	MPI_Barrier(MPI_COMM_WORLD); //synchronise for timing scatter
	scatterTimeStart = MPI_Wtime();
	
	/*
	 *divide work into multiples of world_size, each process handles jobs at its index
	 *e.g say using 8 processes, process 1 handles the 1st column, 9th column ...
	 *when done with your first column, take another column if there is
	 */
	 
	//scatter the elements across processes
	int k,l;
	for(k=0,l=0; k<padded_data_count; k+=PROCESSES,l++){
		MPI_Scatter(inputData+k,1,MPI_INT,processInput+l,1,MPI_INT,0,MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD); //synchronise for timing scatter
	scatterTimeEnd = MPI_Wtime(); //timing
	scatterTimeDuration = scatterTimeEnd - scatterTimeStart; 
	MPI_Reduce(&scatterTimeDuration,&scatterTimeGlobal,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(rank == 0) printf("\nUp scatter completed (Time taken: %.6f)\n\n", scatterTimeGlobal);
	
    
	

	/* DO UP PHASE */
	upTimeStart = MPI_Wtime(); //start up phase timing, synchronisation already done above
	MPI_Request send_req;
	MPI_Request rec_req;
	int columnID, maxNumColumns, rec_rank, send_rank; 
	
	//each process has 2D array to hold outputs, 
	//as the grid is viewed as a matrix of Dimension N by log N //base 2
	int o_row = n_steps+1, o_col = n_inputs; 	// +1 since first row contains given input
	int out[o_col][o_row]; //output from downphase
	
	//storing the given input elements in the first row for processing
	for(k=0; k<n_inputs; k++){
		out[k][0] = *(processInput + k);
	}
	
	//for each column...
	//get the ID for that column, even columns do nothing only send to next in sequence
	for(k=0; k<n_inputs; k++){ //for number of inputs to each process
		columnID = k * PROCESSES + rank; //column ID is [Nth column for that process] * world_size + rank
		maxNumColumns = n_inputs* PROCESSES;
		//if this is an even column, just send to the next columns
		if((columnID+1) % 2 != 0){
		//since the even columns are only sending, then they have direct output for the up phase
			for(l=0; l<o_row; l++){ //for each row in the column
				out[k][l] = out[k][0];
			}
			//get rank to send data to, pass column ID (next column in this case)
			rec_rank = getRank(columnID+1, PROCESSES); 
			//immediate send to next process, will receive when ready, tag is columnId*10+row
			MPI_Isend(&out[k][0],1,MPI_INT,rec_rank,columnID*10,MPI_COMM_WORLD, &send_req); 
		}
		//if this is an odd column, then: receive, update, send
		else {
			//loop through the rows in the column, receive value from 
			//prev offset if value is sent, update next index and send to next offset 
			int stps, offs, nextOffs,prevIdx, nextId;
			for(stps=1; stps<o_row+1; stps++){ //for each row in the column
				offs = (int)pow(2, stps);
				prevIdx = (int)pow(2, stps-1);
				nextOffs = (int)pow(2, stps+1);
				
				//check for column sequence by dividing by powers of 2 i.e 2, 4, 8. ..
				if((columnID+1) % offs == 0){
					//receive index from previous rank
					send_rank = getRank(columnID-prevIdx, PROCESSES); //get sending rank
					//receive, expected tag is: [prev column id] * [10 - prev index], this format is used to tag the up phase
					MPI_Recv(out[k]+stps,1,MPI_INT,send_rank, (columnID-prevIdx)*10+(stps-1) , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//make next index
					out[k][stps] = out[k][stps] + out[k][stps-1]; 
					//if its not divisible by the next offset, then thats its final value for the up phase
					if((columnID+1) % nextOffs != 0){
						for(l=stps; l<o_row; l++){
							out[k][l] = out[k][stps]; //save it as final
						}
					}
					//send next index to the next 2 column which is also the next 2 process if there is a next 2 process
					if((columnID+offs) <= maxNumColumns){
						rec_rank = getRank(columnID+offs, PROCESSES); //get receiving rank, tag is columnId*10+row
						MPI_Isend(&out[k][stps],1,MPI_INT,rec_rank,columnID*10+stps,MPI_COMM_WORLD, &send_req); //send to next column
					} 
				}
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD); //wait until everyone is done.
	upTimeEnd = MPI_Wtime(); //timing 
	upTimeDuration = upTimeEnd - upTimeStart; //timing
	MPI_Reduce(&upTimeDuration,&upTimeGlobal,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); //timing
	//gather the results to outputDataUp
	int p, q;
	for(p=0,q=0; p<o_col; p++, q+=PROCESSES){
		MPI_Gather(&out[p][o_row-1], 1, MPI_INT, outputDataUp+q, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	//print output from upPhase 
	int h;
	if(rank == 0){
		printf("\nUp gather completed.");
		printf("\nOUTPUT FROM UP PHASE\n");
		for(h=0; h<padded_data_count;h++){
			printf(" %d ", *(outputDataUp+h));
		}
		printf("\n"); 
	}
	
	



	
	/* DO DOWN PHASE */ 
	downTimeStart = MPI_Wtime(); //start down phase timing
	MPI_Request send_request;
	
	/*
	 *similar to the up phase, divide work in multiples of World_size
	 *across downphase input buffer, each process handles jobs at its index
	 *e.g say for 8 processes, process 1 handles the 1st element, 9th element ... as columns
	 */
	int cl, ind, oo_row, oo_col; //scatter output from up phase to downphase 
	for(cl=0,ind=0; cl<padded_data_count; cl+=PROCESSES,ind++){
		MPI_Scatter(outputDataUp+cl,1,MPI_INT,processInputDown+ind,1,MPI_INT,0,MPI_COMM_WORLD);
	}
	if(rank == 0) printf("Down scatter completed. \n\n");
	
	//each process hold create 2D array to hold outputs
	oo_row = n_steps, oo_col = n_inputs; // the first row in each process will contain the input to the processes
	int outDown[oo_col][oo_row]; //output from downphase
	
	//initialise the outputs of your column with the input values from 
	//up phase as some columns may not be doing anything until the end
	int ct;
	for(ct=0; ct<n_inputs; ct++){
		for(l=0; l<oo_row; l++){
			outDown[ct][l] = *(processInputDown + ct);
		}
	}
	
	int startOffset = (int)pow(2, n_steps)/2;
	int cOffset, currentRow, rowCount;
	int cntr, colmID, cnter, temp;
	
	//in your column, loop through your rows check if column is to  
	//send or receive, if receiving, receive and update next value
	for(ct=0; ct<n_inputs; ct++){ //for number of input to each process
		colmID = ct * PROCESSES + rank; //get column ID by: [Nth column for that process] * World_size + rank
		maxNumColumns = n_inputs*PROCESSES;
		cOffset = startOffset;
		//in your column, loop through your rows check who is sending or receiving on each row
		for(cOffset=startOffset, currentRow=0; cOffset>1; cOffset/=2, currentRow++){
			// do I have anything to receive
			if(((colmID+1)%cOffset == (cOffset/2)) && (colmID+1 <= padded_data_count) && (colmID+1 >= cOffset)){
				// receive 
				send_rank = getRank(colmID-(cOffset/2), PROCESSES);
				MPI_Recv(outDown[ct]+currentRow+1,1,MPI_INT,send_rank, (colmID-(cOffset/2))*100+currentRow , MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// update values below this row in this column
				temp = outDown[ct][currentRow+1]+outDown[ct][currentRow];
				int stt = currentRow;
				for(stt, cnter=1; stt<oo_row-1; stt++, cnter++)
					outDown[ct][currentRow+cnter] = temp;
			}
			//can I send
			if(((colmID+1)%cOffset == 0) && (colmID+1 <= padded_data_count) && (colmID+1 >= cOffset)){
				// send
				rec_rank = getRank(colmID+(cOffset/2), PROCESSES);
				//send immediately to next column, will receive when ready
				MPI_Isend(&outDown[ct][currentRow],1,MPI_INT,rec_rank,colmID*100+currentRow,MPI_COMM_WORLD, &send_request);
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD); //wait all processes to complete
	downTimeEnd = gatherTimeStart = MPI_Wtime(); //start gather timing
	downTimeDuration = downTimeEnd - downTimeStart; //get duration
	MPI_Reduce(&downTimeDuration,&downTimeGlobal,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);//get time by reduction
	
	//gather the results to outputData
	for(p=0,q=0; p<oo_col; p++, q+=PROCESSES){
		MPI_Gather(&outDown[p][oo_row-1], 1, MPI_INT, outputData+q, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	
	MPI_Barrier(MPI_COMM_WORLD); //synchronise for timing
	gatherTimeEnd = MPI_Wtime(); //end time
	gatherTimeDuration = gatherTimeEnd - gatherTimeStart; //get duration
	MPI_Reduce(&gatherTimeDuration,&gatherTimeGlobal,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); //get time by reduction
	
	//print gather time, overall output and timing for up phase and downphase, timing for serial
	if(rank == 0){
		printf("\nGather completed (Time taken: %.6f)\n", gatherTimeGlobal);
		printf("\nOUTPUT FROM DOWNPHASE\n");
		for(h=0; h<dataCount;h++){
			printf(" %d ", *(outputData+h));
		}
		printf("\n\nUp Phase Time: %.6f\n", upTimeGlobal);
		printf("Down Phase Time: %.6f\n", downTimeGlobal);
		
		//compute serial time
		serialTimeStart = MPI_Wtime();
		int sum = 0;
		for(k=0; k<=dataCount; k++){
			sum += *(inputData + k);
		}
		printf("Sum is: %d\n", sum);
		serialTimeEnd = MPI_Wtime();
		printf("Serial Time: %e\n\n", serialTimeEnd - serialTimeStart);
	}
	
    //free mem
    free(inputData);
	free(outputData);
    free(processInput);
	free(outputDataUp);
	free(processInputDown);
}

//check execution error
int checkInputError(int agc, char *agv[], int world_size) {
    int c = 0;
    //check input data
    if(agc != 2){
		fprintf(stderr, "incorrect argument(s). Input format is: mpiexec -np [world size] ./prefix [datasize] \n");
		return 1;
    } else if((int)atoi(agv[1]) > 10000 || (int)atoi(agv[1]) < 2){
        fprintf(stderr, "invalid argument(s). Number of elements must not be less than 2 or more than 10000 \n");
        return 1;
    } else if(world_size < 4){
		fprintf(stderr, "invalid argument(s). Number of processes must not be less than 4 \n");
        return 1;
	} else if(world_size == 7){
		fprintf(stderr, "invalid argument(s). Number of processes must a multiple of 7 \n");
        return 1;
	}
    return 0;
}

//do the padding, columns are multiples of world_size
int getPaddedCount(int data_count, int worldsize){
	int padded;
	for(padded=2; padded<data_count; padded*=2){
		if(padded > data_count){
			break;
		}
	}
	if(padded == 2 || padded == 4) return worldsize;
	return padded;
}

//get the rank from which to send or receive
int getRank(int columnId, int worldsize){
	return columnId % worldsize;
}
