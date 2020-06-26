package sort.sort;

public class InsertionSort extends SortMain implements IFunctions {
    public InsertionSort() {
        System.out.println("InsertionSort");
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();


        printDiffTime();
    }
}
