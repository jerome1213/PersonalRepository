package sort.sort;

import sort.common.Constant;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SortMain {
    protected int numRandomInteger = 50000;

    protected long beforeTime = 0;
    protected long afterTime = 0;
    protected long secDiffTime = 0;

    protected ArrayList<Integer> mainTestRandomList = Constant.testRandomList;

    public SortMain() {
        Random random = new Random();

        if (mainTestRandomList == null) {
            mainTestRandomList = new ArrayList<>();
            for (int i = 0; i < numRandomInteger; i++) {
                mainTestRandomList.add(random.nextInt(numRandomInteger));
            }
            Constant.testRandomList = mainTestRandomList;

            System.out.println("===== Created Random Integer Array =====");
            System.out.println(mainTestRandomList);
            System.out.println("========================================\n");
        }
    }

    public void printDiffTime() {
        afterTime = System.currentTimeMillis();
        secDiffTime = (afterTime - beforeTime);
        System.out.println("소요시간 : " + secDiffTime + "ms\n");
    }

    protected int getCenterPosition(int start, int end) {
        int center = (start + end) / 2;
        if ((start + end) % 2 == 1) {
            center = (start + (end - 1)) / 2;
        }
        return center;
    }

    public void swap(ArrayList<Integer> array, int a, int b) {
        int temp = array.get(a);
        array.set(a, array.get(b));
        array.set(b, temp);
    }
}
