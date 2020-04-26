/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class PassInfo implements Serializable {
    public double l2_decay_loss;
    public double l1_decay_loss;
    public double cost_loss;
    public double softmax_loss;
    public double loss;
}
