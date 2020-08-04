//Copyright (c) 2011 ashelly.myopenid.com under <http://w...content-available-to-author-only...e.org/licenses/mit-license>

#include <median_filt.h>

//returns 1 if heap[i] < heap[j]
int MedianFilter::mmless(int i, int j)
{
    return ItemLess(data[heap[i]], data[heap[j]]);
}

//swaps items i&j in heap, maintains indexes
int MedianFilter::mmexchange(int i, int j)
{
    int t = heap[i];
    heap[i] = heap[j];
    heap[j] = t;
    pos[heap[i]] = i;
    pos[heap[j]] = j;
    return 1;
}

//swaps items i&j if i<j;  returns true if swapped
int MedianFilter::mmCmpExch(int i, int j)
{
    return (mmless(i, j) && mmexchange(i, j));
}

//maintains minheap property for all items below i/2.
void MedianFilter::minSortDown(int i)
{
    for (; i <= minCt(ct); i *= 2)
    {
        if (i > 1 && i < minCt(ct) && mmless(i + 1, i)) { ++i; }
        if (!mmCmpExch(i, i / 2)) { break; }
    }
}

//maintains maxheap property for all items below i/2. (negative indexes)
void MedianFilter::maxSortDown(int i)
{
    for (; i >= -maxCt(ct); i *= 2)
    {
        if (i<-1 && i > -maxCt(ct) && mmless(i, i - 1)) { --i; }
        if (!mmCmpExch(i / 2, i)) { break; }
    }
}

//maintains minheap property for all items above i, including median
//returns true if median changed
int MedianFilter::minSortUp(int i)
{
    while (i > 0 && mmCmpExch(i, i / 2)) i /= 2;
    return (i == 0);
}

//maintains maxheap property for all items above i, including median
//returns true if median changed
int  MedianFilter::maxSortUp(int i)
{
    while (i < 0 && mmCmpExch(i / 2, i))  i /= 2;
    return (i == 0);
}

/*--- Public Interface ---*/


//creates new Mediator: to calculate `nItems` running median. 
//mallocs single block of memory, caller must free.
MedianFilter::MedianFilter(int nItems)
{
    data.reserve(nItems);
    pos = new int[nItems];
    allocatedHeap = new int[nItems];
    heap = allocatedHeap + (nItems / 2); //points to middle of storage.
    N = nItems;
    ct = idx = 0;
    while (nItems--)  //set up initial heap fill pattern: median,max,min,max,...
    {
        pos[nItems] = ((nItems + 1) / 2) * ((nItems & 1) ? -1 : 1);
        heap[pos[nItems]] = nItems;
    }
    
}

void MedianFilter::Init(int nItems)
{
    data.reserve(nItems);
    pos = new int[nItems];
    allocatedHeap = new int[nItems];
    heap = allocatedHeap + (nItems / 2); //points to middle of storage.
    N = nItems;
    ct = idx = 0;
    while (nItems--)  //set up initial heap fill pattern: median,max,min,max,...
    {
        pos[nItems] = ((nItems + 1) / 2) * ((nItems & 1) ? -1 : 1);
        heap[pos[nItems]] = nItems;
    }

}

//Inserts item, maintains median in O(lg nItems)
void MedianFilter::Insert(float v)
{
    int isNew = (ct < N);
    int p = pos[idx];
    float old = data[idx];
    data[idx] = v;
    idx = (idx + 1) % N;
    ct += isNew;
    if (p > 0)         //new item is in minHeap
    {
        if (!isNew && ItemLess(old, v)) {minSortDown(p * 2); }
        else if (minSortUp(p)) { maxSortDown(-1); }
    }
    else if (p < 0)   //new item is in maxheap
    {
        if (!isNew && ItemLess(v, old)) { maxSortDown(p * 2); }
        else if (maxSortUp(p)) { minSortDown(1); }
    }
    else            //new item is at median
    {
        if (maxCt(ct)) { maxSortDown(-1); }
        if (minCt(ct)) { minSortDown(1); }
    }
}

//returns median item (or average of 2 when item count is even)
float MedianFilter::Median()
{
    float v = data[heap[0]];
    if ((ct & 1) == 0) { v = ItemMean(v, data[heap[-1]]); }
    return v;
}


/*--- Test Code ---*/
#include <cstdio>
void MedianFilter::PrintMaxHeap()
{
    int i;
    if (maxCt(ct))
        printf("Max: %3f", data[heap[-1]]);
    for (i = 2; i <= maxCt(ct); ++i)
    {
        printf("|%3f ", data[heap[-i]]);
        if (++i <= maxCt(ct)) printf("%3f", data[heap[-i]]);
    }
    printf("\n");
}

void MedianFilter::PrintMinHeap()
{
    int i;
    if (minCt(ct))
        printf("Min: %3f", data[heap[1]]);
    for (i = 2; i <= minCt(ct); ++i)
    {
        printf("|%3f ", data[heap[i]]);
        if (++i <= minCt(ct)) printf("%3f", data[heap[i]]);
    }
    printf("\n");
}

void MedianFilter::ShowTree()
{
    PrintMaxHeap();
    printf("Mid: %3f\n", data[heap[0]]);
    PrintMinHeap();
    printf("\n");
}

int main(int argc, char* argv[])
{
    int i;
    float v;
    MedianFilter m[18];
    for(int j=0;j < 18;j++)
        m[j].Init(15);

    std::cout << "Done!\n";
    for (i = 0; i < 30; i++)
    {
        v = i + 0.5; // rand() & 127;
        printf("Inserting %3f \n", v);
        m[0].Insert(v);
        v = m[0].Median();
        printf("Median = %3f.\n\n", v);
        m[0].ShowTree();
    }
}