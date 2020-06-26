package sort.sort;

public class SelectionSort extends SortMain implements IFunctions {

    public SelectionSort() {
        System.out.println("SelectionSort");
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        for (int i = 0; i < numRandomInteger - 1; i++) {
            int min = i;
            for (int j = i + 1; j < numRandomInteger; j++) {
                if (testRandomList.get(min) > testRandomList.get(j)) {
                    min = j;
                }
            }
            swap(min, i);
        }

        printDiffTime();
    }
}
