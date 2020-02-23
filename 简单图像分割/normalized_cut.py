#快速排序
class QuickSort:
    def __init__(self,array):
        self.array = array
        length = len(self.array)
        self.quicksort(0,length-1)
    
    def quicksort(self,startindex,endindex):
        if startindex >= endindex:
            return
        pivotindex = self.partition(startindex,endindex)
        self.quicksort(startindex,pivotindex-1)
        self.quicksort(pivotindex+1,endindex)
        return self.array
    
    def partition(self,startindex,endindex):
        m,n = startindex,startindex+1 #双指针,m慢n快
        pivot = self.array[startindex]
        while n <= endindex:
            if self.array[n]>pivot:
                n += 1
            else:
                m += 1
                self.array[m],self.array[n] = self.array[n],self.array[m]
                n += 1
        self.array[startindex],self.array[m] = self.array[m],self.array[startindex]
        return m
array = [5,3,1,7,9,2,4,6,8,10,109,23,1,54,6]
QuickSort(array)
print(array)