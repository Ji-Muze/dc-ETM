import torch
import torch.nn as nn
from torch.utils import tensorboard
from torch.nn.parameter import Parameter

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model_dcetm import DCETM
from model_dcetm_beta import beta_DCETM
import utils
from gensim.matutils import corpus2csc
from gensim.models.coherencemodel import CoherenceModel
from scipy.sparse import coo_matrix
import pickle
import time

tqdm.disable= True

class DeepCoupling_trainer(object):
    def __init__(self, args, voc_path='voc.txt'):
        super(DeepCoupling_trainer, self).__init__()
        self.args = args
        self.save_path = args.save_path

        self.discount = args.discount
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.save_phi_every_epoch = 1000
        dataset_dir = self.args.dataset_dir
        # if dataset_dir[-8:] == '20ng.pkl':
        #     with open('./dataset/20ng.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-6:] == 'R8.pkl':
        #     with open('./dataset/R8.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-8:] == 'zhdd.pkl':
        #     with open('./dataset/zhdd.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-9:] == 'trump.pkl':
        #     with open('./dataset/trump.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-15:] == 'threetopics.pkl':
        #     with open('./dataset/threetopics.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        with open(dataset_dir, 'rb') as f:
            dataset = pickle.load(f)
        labels = dataset['label']
        self.count = len(labels)

        # model
        if args.use_beta:
            self.model = beta_DCETM(args)
            self.decoder_optimizer = torch.optim.Adam([{'params': self.model.decoder.parameters()},
                                                       {'params': self.model.betaDecoder.parameters()}], lr=self.lr,
                                                      weight_decay=self.weight_decay)
        else:
            self.model = DCETM(args)
            self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr,
                                                      weight_decay=self.weight_decay)

        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.layer_alpha_optimizer = torch.optim.Adam([self.model.layer_alpha], lr=self.lr * 0.01, weight_decay=self.weight_decay)

        # log
        self.writer = tensorboard.SummaryWriter(os.path.join(args.save_path, "tf_log"))
        self.log_path = os.path.join(args.save_path, "log.txt")
        utils.add_show_log(self.log_path, str(args))


    def train(self, train_data_loader, test_data_loader=None):
        total_time = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            utils.add_show_log(self.log_path, "======================== train ========================")
            utils.add_show_log(self.log_path, f"epoch {epoch:03d}|{self.epochs:03d}:")

            self.model.to(self.args.device)
            KL_batch = [0] * (self.layer_num)
            likelihood_batch = [0] * (self.layer_num)
            division_likeli_loss_batch = 0.0
            num_data = len(train_data_loader)

            for train_data, train_label in tqdm.tqdm(train_data_loader):
                train_data = train_data.to(self.args.device)

                # update inference network
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                ret_dict = self.model(train_data)
                KL_loss = ret_dict['loss'][1:]
                Likelihood = ret_dict['likelihood'][1:]

                Q_value = torch.tensor(0., device=self.args.device)
                for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                    Q_value += 10. * (Likelihood[t] + KL_loss[t])
                Q_value.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.layer_alpha_optimizer.step()
                    self.layer_alpha_optimizer.zero_grad()
                    self.optimizer.zero_grad()

                # update generative network
                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                ret_dict = self.model(train_data)
                KL_loss = ret_dict['loss'][1:]
                Likelihood = ret_dict['likelihood'][1:]

                Q_value = torch.tensor(0., device=self.args.device)
                for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                    Q_value += 10. * (Likelihood[t] + KL_loss[t])
                Q_value.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.layer_alpha_optimizer.step()
                    self.layer_alpha_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()

                # update alpha
                self.model.train()  # require_grad for alpha

                ret_dict = self.model(train_data)

                division_loss = ret_dict["division_likeli_loss"]
                division_loss.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.layer_alpha_optimizer.step()
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()
                    self.layer_alpha_optimizer.zero_grad()
                    self.optimizer.zero_grad()

                division_likeli_loss_batch += ret_dict['division_likeli_loss'].item() / num_data

                for t in range(self.layer_num):
                    KL_batch[t] += ret_dict['loss'][t + 1].item() / num_data
                    likelihood_batch[t] += ret_dict['likelihood'][t + 1].item() / num_data


            # evaluate
            if epoch % self.args.eval_epoch_num == 0:
                # write to the file
                activated_layer_alpha = torch.nn.functional.sigmoid(self.model.layer_alpha)
                softmax_layer_alpha = torch.nn.functional.softmax(activated_layer_alpha, dim=0)
                print(f'softmax_layer_alpha: {softmax_layer_alpha}')
                for t in range(self.layer_num):
                    log_str = 'epoch {}|{}, layer {}|{}, kl: {}, likelihood: {}, devision_likelihood: {}'.format(epoch, self.epochs, t,
                                                                                              self.layer_num,
                                                                                              KL_batch[t],
                                                                                              likelihood_batch[t],
                                                                                              division_likeli_loss_batch)
                    utils.add_show_log(self.log_path, log_str)

                    self.writer.add_scalar(f"train/kl_loss{t}", KL_batch[t], epoch)
                    self.writer.add_scalar(f"train/likelihood_{t}", likelihood_batch[t], epoch)
                self.writer.add_scalar(f"train/division_likelihood", division_likeli_loss_batch, epoch)


            if epoch % self.args.test_epoch_num == 0 and self.args.ppl:
                print(f"PPLtest:{self.args.ppl}")
                if self.args.clustering:
                    print(f"CLUStest:{self.args.clustering}")
                    pass
                else:
                    print("Do TEST")
                    self.test(train_data_loader, epoch)

            # if epoch % self.args.save_epoch_num == 0:
            if epoch % 100 == 0:
                model_path = os.path.join(self.args.save_path, "model")
                file_name = model_path + '/dcetm_' + self.args.task + str(epoch) + '.pt'
                utils.save_checkpoint({'state_dict': self.model.state_dict()}, file_name, True)
            
            end_time = time.time()
            epoch_time = end_time - start_time
            #print(f"Epoch {epoch} took {epoch_time:.2f} seconds")
            total_time += epoch_time
            utils.add_show_log(self.log_path, f"epoch {epoch} took {epoch_time:.2f} seconds")

        avg_time = total_time / self.epochs
        utils.add_show_log(self.log_path, f"Train finished")
        utils.add_show_log(self.log_path, f"Average time per epoch: {avg_time:.2f} seconds")

    def test(self, data_loader, epoch):
        utils.add_show_log(self.log_path, "======================== test ========================")

        # test
        self.model.eval()
        num_data = len(data_loader)
        KL_batch = [0] * (self.layer_num)
        PPL_batch = [0] * (self.layer_num)
        likelihood_batch = [0] * (self.layer_num)
        division_PPL_batch = 0.0
        division_likeli_loss_batch = 0.0

        for test_data, test_label in tqdm.tqdm(data_loader):
            test_data = test_data.to(self.args.device)
            test_label = test_label.to(self.args.device)

            ret_dict = self.model.test_ppl(test_data, test_label)
            PPL_minibatch = ret_dict["ppl"]
            division_PPL_batch += ret_dict["ppl"][0].item() / num_data

            ret_dict = self.model.forward(test_label)
            division_likeli_loss_batch += ret_dict["division_likeli_loss"].item() / num_data

            for t in range(self.layer_num):
                KL_batch[t] += ret_dict['loss'][t + 1].item() / num_data
                likelihood_batch[t] += ret_dict['likelihood'][t + 1].item() / num_data
                PPL_batch[t] += PPL_minibatch[t + 1].item() / num_data


        # write to the file
        for t in range(self.layer_num):
            log_str = 'epoch {}|{}, layer {}|{}, ppl:{}, devision_ppl: {}, likelihood: {}, devision_likelihood: {}'.format(epoch, self.epochs,
                                                                                                                 t, self.layer_num,
                                                                                                                 PPL_batch[t],
                                                                                                                 division_PPL_batch,
                                                                                                         likelihood_batch[t],
                                                                                                         division_likeli_loss_batch)
            utils.add_show_log(self.log_path, log_str)

            self.writer.add_scalar(f"test/kl_loss{t}", KL_batch[t], epoch)
            self.writer.add_scalar(f"test/likelihood_{t}", likelihood_batch[t], epoch)
            self.writer.add_scalar(f"test/PPL_{t}", PPL_batch[t], epoch)
        self.writer.add_scalar(f"test/devision_ppl", division_PPL_batch, epoch)
        self.writer.add_scalar(f"test/division_likelihood", division_likeli_loss_batch, epoch)


    def train_for_clustering(self, train_data_loader):
        self.model.to(self.args.device)
        KL_batch = [0] * (self.layer_num)
        likelihood_batch = [0] * (self.layer_num)
        division_likeli_loss_batch = 0.0
        num_data = len(train_data_loader)

        for train_data, train_label in tqdm.tqdm(train_data_loader):
            train_data = train_data.to(self.args.device)
            # self.optimizer.zero_grad()
            # self.decoder_optimizer.zero_grad()
            # self.layer_alpha_optimizer.zero_grad()

            # update inference network
            self.model.h_encoder.train()
            self.model.shape_encoder.train()
            self.model.scale_encoder.train()
            self.model.decoder.eval()

            ret_dict = self.model(train_data)
            #keys = list(ret_dict.keys())
            #print("ret_dict keys: ", keys)
            KL_loss = ret_dict['loss'][1:]
            Likelihood = ret_dict['likelihood'][1:]
            #print("KL_loss: ", KL_loss)
            #print("Likelihood: ", Likelihood)

            Q_value = torch.tensor(0., device=self.args.device)
            for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                Q_value += 10. * (Likelihood[t] + KL_loss[t])
            Q_value.backward()

            # Q_value = 10 * (ret_dict['division_likeli_loss'] + Likelihood[0])
            # for t in range(self.layer_num - 1):
            #     Q_value += 10 * self.args.kl_weight*KL_loss[t]
            # Q_value.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                self.optimizer.step()
                self.layer_alpha_optimizer.step()
                self.layer_alpha_optimizer.zero_grad()
                self.optimizer.zero_grad()

            # update generative network
            self.model.h_encoder.eval()
            self.model.shape_encoder.eval()
            self.model.scale_encoder.eval()
            self.model.decoder.train()

            ret_dict = self.model(train_data)
            KL_loss = ret_dict['loss'][1:]
            Likelihood = ret_dict['likelihood'][1:]

            Q_value = torch.tensor(0., device=self.args.device)
            for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                Q_value += 10. * (Likelihood[t] + KL_loss[t])
            Q_value.backward()

            # Q_value = 10 * (ret_dict['division_likeli_loss'] + Likelihood[0])
            # for t in range(self.layer_num - 1):
            #     Q_value += 10 * self.args.kl_weight*KL_loss[t]
            # Q_value.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                self.decoder_optimizer.step()
                self.layer_alpha_optimizer.step()
                self.layer_alpha_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

            # update alpha
            self.model.train()  # require_grad for alpha
            # self.model.h_encoder.eval()
            # self.model.shape_encoder.eval()
            # self.model.scale_encoder.eval()
            # self.model.decoder.eval()

            ret_dict = self.model(train_data)

            division_loss = ret_dict["division_likeli_loss"]
            division_loss.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                self.optimizer.step()
                self.layer_alpha_optimizer.step()
                self.decoder_optimizer.step()
                self.decoder_optimizer.zero_grad()
                self.layer_alpha_optimizer.zero_grad()
                self.optimizer.zero_grad()

    def load_model(self, save_path):
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.args.device)

    def extract_theta(self, data_loader):
        self.model.eval()
        test_theta_batch = []
        test_label_batch = []
        for test_data, test_label in tqdm.tqdm(data_loader):
            test_data = test_data.type(torch.float).to(self.args.device)
            test_label = test_label.to(self.args.device)
            ret_dict = self.model.forward(test_data)
            test_theta_batch.append([ret_dict['theta'][i].detach().cpu().numpy() for i in range(self.layer_num)])
            test_label_batch.append(test_label.detach().cpu().numpy())

        return test_theta_batch, test_label_batch

    def vis(self):
        # layer1
        #w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        w_1 = torch.mm(self.model.decoder[0].rho, torch.transpose(self.model.decoder[0].alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            #tmp = dic1[:, i].reshape(28, 28)
            tmp = dic1[:, i].reshape(50, 40)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        #w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        w_2 = torch.mm(self.model.decoder[1].rho, torch.transpose(self.model.decoder[1].alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        # dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            #tmp = dic2[:, i].reshape(28, 28)
            tmp = dic2[:, i].reshape(50, 40)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        #w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        w_3 = torch.mm(self.model.decoder[2].rho, torch.transpose(self.model.decoder[2].alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        # dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])
        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            #tmp = dic3[:, i].reshape(28, 28)
            tmp = dic3[:, i].reshape(50, 40)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath, top_n=10):
        if self.voc is not None:
            print(self.args.dataset_dir)
            dataset_dir = self.args.dataset_dir
            if dataset_dir[-8:] == '20ng.pkl':
                dataset_path = './freqdata/20ngnew/'
            if dataset_dir[-6:] == 'R8.pkl':
                dataset_path = './freqdata/R8/'
            if dataset_dir[-8:] == 'zhdd.pkl':
                dataset_path = './freqdata/zhdd/'
            if dataset_dir[-9:] == 'trump.pkl':
                dataset_path = './freqdata/trump/'
            if dataset_dir[-15:] == 'threetopics.pkl':
                dataset_path = './freqdata/threetopics/'
            if dataset_dir[-10:] == 'sample.pkl':
                dataset_path = './freqdata/dbpedia_sample/'
            if dataset_dir[-14:] == 'googlenews.pkl':
                dataset_path = './freqdata/googlenews/'
            if dataset_dir[-12:] == 'Snippets.pkl':
                dataset_path = './freqdata/SearchSnippets/'
            if dataset_dir[-13:] == 'TagMyNews.pkl':
                dataset_path = './freqdata/TagMyNews/'
            if dataset_dir[-9:] == 'Tweet.pkl':
                dataset_path = './freqdata/Tweet/'
            if dataset_dir[-10:] == 'agnews.pkl':
                dataset_path = './freqdata/agnews/'

            with open(dataset_path + 'co_occurrence_matrix.pkl', 'rb') as f:
                co_occurrence_matrix = pickle.load(f)

            with open(dataset_path + 'row_sums.pkl', 'rb') as f:
                row_sums = pickle.load(f)

            with open(dataset_path + 'word_frequencies.pkl', 'rb') as f:
                word_times = pickle.load(f)
            #print("how many numbers in word_times:", len(word_times))
            utils.chk_mkdir(outpath)
            phi = 1
            all_topic_words = []
            #for num, phi_layer in enumerate(Phi):
            for num, phi_layer in enumerate(Phi[:1]):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words, top_n_words2, top_n_freqs = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                    # f.write(str(top_n_freqs))
                    # f.write('\n')
                    all_topic_words.append(top_n_words2)
                    del top_n_words
                    del top_n_freqs
                f.close()
                all_topic_words_flat = [word for topic_words in all_topic_words for word in topic_words]
                #print(len(all_topic_words_flat))
                word_count = {}
                for word in all_topic_words_flat:
                    word_count[word] = word_count.get(word, 0) + 1
                unique_words = [word for word, count in word_count.items() if count == 1]

                freq_path = os.path.join(outpath, 'freqs' + str(num) + '.txt')
                freqs_file = open(freq_path,'w')
                # 计算每个主题的TC和TD值
                topic_coherence = []
                topic_diversity = []
                topic_quality = []
                for topic_id in range(phi_k):
                    #print(topic_id)
                    top_n_words = []
                    top_n_freqs = []
                    freqs_file.write(f"Topic: {topic_id}\n")
                    word_useless, top_n_words, top_n_freqs = self.get_top_n(phi[:, topic_id], top_n)
                    #print(top_n_words)
                    top_n_indices = []
                    for word in top_n_words:
                        if word in self.voc:
                            top_n_indices.append(self.voc.index(word))
                        else:
                            top_n_indices.append(None)

                    unique_words_count = 0
                    for word in top_n_words:
                        if word in unique_words:
                            unique_words_count += 1

                    tc = 0
                    for i in range(len(top_n_words)):
                        freqs_file.write(f"Word {i}\n")
                        for j in range(i + 1, len(top_n_words)):
                            if top_n_indices[i] < top_n_indices[j]:
                                co_occurrence = co_occurrence_matrix.toarray()[top_n_indices[i]][top_n_indices[j]]
                                #co_occurrence_freq = co_occurrence / row_sums[top_n_indices[i]]
                            else :
                                co_occurrence = co_occurrence_matrix.toarray()[top_n_indices[j]][top_n_indices[i]]
                                #co_occurrence_freq = co_occurrence / row_sums[top_n_indices[j]]
                            co_occurrence_freq = co_occurrence / self.count
                            co_freq = np.squeeze(co_occurrence_freq)
                            co_freq = float(co_freq)
                            freqs_file.write(f"co_freq of word '{top_n_words[i]}' and '{top_n_words[j]}' is: {co_freq}\n")
                            #print("co_freq of two words: ", co_freq)
                            ind_i = self.find_index(self.voc, top_n_words[i])
                            ind_j = self.find_index(self.voc, top_n_words[j])
                            freq_i = word_times[ind_i] / self.count
                            freq_i = np.squeeze(freq_i)
                            freq_i = float(freq_i)
                            freq_j = word_times[ind_j] / self.count
                            freq_j = np.squeeze(freq_j)
                            freq_j = float(freq_j)
                            if co_freq == 0:
                                npmi = 0
                            else :
                                pmi = np.log( co_freq / (freq_i * freq_j) )
                                npmi = - pmi / np.log(co_freq)

                            tc += npmi
                        #print("freq from dataset: ", freq_i)
                        freqs_file.write(f"freq from dataset: {freq_i}\n")
                        #print("freq from model phi: ", top_n_freqs[i])
                        freqs_file.write(f"freq from model phi: {top_n_freqs[i]}\n")
                    tc /= (top_n * (top_n - 1) / 2)
                    td = unique_words_count / top_n
                    tq = tc * td

                    topic_coherence.append(tc)
                    topic_diversity.append(td)
                    topic_quality.append(tq)

                    print(f"Topic {topic_id + 1}: TC = {tc}, TD = {td}, TQ = {tq} ")
                

                freqs_file.close()
                # 将TC和TD值写入文件
                tc_path = os.path.join(outpath, f"topic_coherence_{num}.txt")
                td_path = os.path.join(outpath, f"topic_diversity_{num}.txt")
                tq_path = os.path.join(outpath, f"topic_quality_{num}.txt")
                np.savetxt(tc_path, topic_coherence, fmt="%.4f")
                np.savetxt(td_path, topic_diversity, fmt="%.4f")
                np.savetxt(tq_path, topic_quality, fmt="%.4f")
                tc_avg = np.mean(topic_coherence)
                td_avg = np.mean(topic_diversity)
                tq_avg = tc_avg * td_avg
                print(f"Result of this epoch: TC: {tc_avg}, TD: {td_avg}, TQ: {tq_avg}")
                del topic_coherence
                del topic_diversity
                del topic_quality
                
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        top_n_words2 = []
        top_n_freqs = []
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
            top_n_words2.append(self.voc[index])
            top_n_freqs.append(phi[index])
        return top_n_words, top_n_words2, top_n_freqs

    def vis_txt(self, epoch):

        phi = []
        for t in range(self.layer_num):
            phi.append(self.model.decoder[t].get_phi().cpu().detach().numpy())
            # w_t = torch.mm(self.model.decoder[t].rho, torch.transpose(self.model.decoder[t].alphas, 0, 1))
            # phi_t = torch.softmax(w_t, dim=0).cpu().detach().numpy()
            # phi.append(phi_t)
        if epoch % self.save_phi_every_epoch == 0:
            print("Epoch: {}".format(epoch))
            self.vision_phi(phi, os.path.join(self.save_path, "phi", f"{epoch:03d}"))
        # self.vision_phi(phi, os.path.join(self.save_path, "phi", "latest"))

    def find_index(self, arr, target):
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1