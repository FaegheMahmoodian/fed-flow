import threading

from torch import optim, nn

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.base_model import BaseModel
from app.dto.message import IterationFlagMessage, GlobalWeightMessage, SplitLayerConfigMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser
from app.model.utils import get_available_torch_device
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class FedEdgeServer(FedBaseNodeInterface):

    def __init__(self, ip: str, port: int, model_name, dataset, offload, aggregator: BaseAggregator,
                 neighbors: list[NodeIdentifier]):
        super().__init__(ip, port, NodeType.EDGE, '', neighbors)
        self._edge_based = None
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.nets = {}
        self.group_labels = None
        self.criterion = nn.CrossEntropyLoss()
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.central_server_communicator = Communicator()
        self.offload = offload
        self.aggregator = aggregator

        if offload:
            model_len = model_utils.get_unit_model_len()
            self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device, self.is_edge_based)

            self.testset = data_utils.get_testset()
            self.testloader = data_utils.get_testloader(self.testset, 0)
        self.neighbor_bandwidth: dict[NodeIdentifier, BandWidth] = {}
        self.optimizers = None
        self.split_layers = {}

    @property
    def is_edge_based(self) -> bool:
        if self._edge_based is not None:
            return self._edge_based
        server_neighbors = self.get_neighbors([NodeType.SERVER])
        self._edge_based = len(server_neighbors) > 0
        return self._edge_based

    def initialize(self, learning_rate):
        self.nets = {}
        self.optimizers = {}
        self.scheduler = {}
        if not self.is_edge_based:
            self.initialize_split_layers()
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]
            if split_point < len(self.uninet.cfg) - 1:
                self.nets[neighbor] = model_utils.get_model('Edge', self.split_layers[neighbor], self.device,
                                                            self.is_edge_based)
                cweights = model_utils.get_model('Client', self.split_layers[neighbor], self.device,
                                                 self.is_edge_based).state_dict()
                pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                            self.nets[neighbor].state_dict(), [])
                self.nets[neighbor].load_state_dict(pweights)

                if len(list(self.nets[neighbor].parameters())) != 0:
                    self.optimizers[neighbor] = optim.SGD(self.nets[neighbor].parameters(), lr=learning_rate,
                                                          momentum=0.9, weight_decay=5e-4)
                    self.scheduler[neighbor] = optim.lr_scheduler.StepLR(self.optimizers[neighbor],
                                                                         config.lr_step_size, config.lr_gamma)
            else:
                self.nets[neighbor] = model_utils.get_model('Edge', self.split_layers[neighbor], self.device,
                                                            self.is_edge_based)

    def gather_and_scatter_global_weight(self):
        received_messages = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.SERVER])
        msg: GlobalWeightMessage = received_messages[0].message
        weights = msg.weights[0]
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            cweights = model_utils.get_model('Client', self.split_layers[neighbor], self.device,
                                             self.is_edge_based).state_dict()
            pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                            self.nets[neighbor].state_dict())
            self.nets[neighbor].load_state_dict(pweights)
        self.scatter_msg(GlobalWeightMessage([weights]), [NodeType.CLIENT])

    def clustering(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))(self)

    def get_neighbors_bandwidth(self) -> dict[NodeIdentifier, BandWidth]:
        return self.neighbor_bandwidth

    def split(self, state, options: dict):
        result = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels, self)

        # --- DEBUG FIX: ensure dict[NodeIdentifier -> cut] for decentralized client-edge ---
        if isinstance(result, list):
            clients = list(self.get_neighbors([NodeType.CLIENT]))
            default_cut = model_utils.get_unit_model_len() - 1
            mapped = {}
            for i, c in enumerate(clients):
                cut = result[i] if i < len(result) else default_cut
                # clamp to valid range
                cut = max(1, min(int(cut), default_cut))
                mapped[c] = cut
            result = mapped

        self.split_layers = result
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def gather_and_scatter_split_config(self):
        received_messages = self.gather_msgs(SplitLayerConfigMessage.MESSAGE_TYPE, [NodeType.SERVER])
        msg: SplitLayerConfigMessage = received_messages[0].message
        self.split_layers = msg.data
        self.scatter_split_layers([NodeType.CLIENT])

    def start_decentralized_training(self):
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(target=self._thread_decentralized_training,
                                                      args=(neighbor,), name=str(neighbor))
            fed_logger.info(str(neighbor) + ' offloading training start')
            self.threads[neighbor].start()

        fed_logger.info('waiting for offloading training to finish')
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info('offloading training finished')

    def _thread_decentralized_training(self, neighbor: NodeIdentifier):
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        flag: bool = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                   IterationFlagMessage.MESSAGE_TYPE).flag
        while flag:
            flag = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                 IterationFlagMessage.MESSAGE_TYPE).flag
            if not flag:
                break
            msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[neighbor] < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers.keys():
                    self.optimizers[neighbor].zero_grad()
            outputs = self.nets[neighbor](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[neighbor] < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers:
                    self.optimizers[neighbor].step()
                    self.scheduler[neighbor].step()

            fed_logger.info(str(neighbor) + " sending gradients")
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)

        fed_logger.info(str(neighbor) + ' offloading training end')

    def start_centralized_training(self):
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(target=self._thread_centralized_training,
                                                      args=(neighbor,), name=str(neighbor))
            fed_logger.info(str(neighbor) + ' offloading training start')
            self.threads[neighbor].start()

        fed_logger.info('waiting for offloading training to finish')
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info('offloading training finished')

    def _thread_centralized_training(self, neighbor: NodeIdentifier):
        self._forward_propagation(neighbor)
        self._send_back_local_weight(neighbor)

    def _forward_propagation(self, neighbor: NodeIdentifier):
        msg: IterationFlagMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                  IterationFlagMessage.MESSAGE_TYPE)
        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]
        edge_exchange = self.get_exchange_name(neighbor)
        flag: bool = msg.flag
        if self.split_layers[neighbor][1] < model_utils.get_unit_model_len() - 1:
            self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                          IterationFlagMessage(flag))
        else:
            self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                          IterationFlagMessage(False))

        while flag:
            if self.split_layers[neighbor][0] < model_utils.get_unit_model_len() - 1:
                msg: IterationFlagMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                          IterationFlagMessage.MESSAGE_TYPE)
                flag: bool = msg.flag

                if not flag:
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  IterationFlagMessage(flag))
                    break

                msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(),
                                                         config.current_node_mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                smashed_layers = msg.weights[0]
                labels = msg.weights[1]

                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if self.split_layers[neighbor][0] < self.split_layers[neighbor][1]:
                    if neighbor in self.optimizers.keys():
                        self.optimizers[neighbor].zero_grad()
                    outputs = self.nets[neighbor](inputs)
                    if self.split_layers[neighbor][1] < model_utils.get_unit_model_len() - 1:
                        self.send_msg(edge_exchange,
                                      HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                      IterationFlagMessage(flag))
                        msg: list = [outputs.to(self.device), targets.to(self.device)]
                        self.send_msg(edge_exchange,
                                      HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                      GlobalWeightMessage(msg))
                        msg: GlobalWeightMessage = self.recv_msg(edge_exchange,
                                                                 config.current_node_mq_url,
                                                                 GlobalWeightMessage.MESSAGE_TYPE)
                        gradients = msg.weights[0].to(self.device)
                        outputs.backward(gradients)
                        msg: list = [inputs.grad]
                        self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor),
                                      GlobalWeightMessage(msg))
                    else:
                        outputs = self.nets[neighbor](inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        if neighbor in self.optimizers.keys():
                            self.optimizers[neighbor].step()
                        msg: list = [inputs.grad]
                        self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor),
                                      GlobalWeightMessage(msg))
                else:
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  IterationFlagMessage(flag))
                    msg: list = [inputs.cpu(), targets.cpu()]
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  GlobalWeightMessage(msg))
                    msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(),
                                                             config.current_node_mq_url,
                                                             GlobalWeightMessage.MESSAGE_TYPE)
                    self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor), msg)
        fed_logger.info(str(neighbor) + ' offloading training end')

    def _send_back_local_weight(self, neighbor: NodeIdentifier):
        cweights = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                 GlobalWeightMessage.MESSAGE_TYPE).weights[0]
        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]
        split_point = self.split_layers[neighbor][0]
        if split_point != (config.model_len - 1):
            w_local = model_utils.concat_weights(self.uninet.state_dict(), cweights,
                                                 self.nets[neighbor].state_dict())
        else:
            w_local = cweights
        msg = GlobalWeightMessage([w_local])
        self.send_msg(self.get_exchange_name(neighbor), HTTPCommunicator.get_rabbitmq_url(server_neighbor), msg)

    def gather_local_weights(self) -> dict[str, BaseModel]:
        client_local_weights = {}
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            client_local_weights[neighbor] = msg.weights[0]
        return client_local_weights

    def aggregate(self, client_local_weights: dict[str, BaseModel]) -> None:
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        w_local_list = self._concat_neighbor_local_weights(client_local_weights)
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)

    def _concat_neighbor_local_weights(self, client_local_weights) -> list:
        w_local_list = []
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            split_point = self.split_layers[neighbor]
            w_local = (client_local_weights[neighbor], config.N / len(client_neighbors))
            if self.offload and split_point != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), client_local_weights[str(neighbor)],
                                               self.nets[neighbor].state_dict()),
                    config.N / len(client_neighbors))
            w_local_list.append(w_local)
        return w_local_list

    def bandwidth(self) -> dict[NodeIdentifier, BandWidth]:
        return self.neighbor_bandwidth

    def gossip_with_neighbors(self):
        edge_neighbors = self.get_neighbors([NodeType.EDGE])
        if len(edge_neighbors) == 0:
            return
        msg = GlobalWeightMessage([self.uninet.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])
        gathered_msgs = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
        gathered_models = [(msg.message.weights[0], config.N / len(edge_neighbors)) for msg in gathered_msgs]
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, gathered_models)
        self.uninet.load_state_dict(aggregated_model)
