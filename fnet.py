from torch import nn

from dataset_loader import *


class FNET(nn.Module):
    def __init__(self):
        super(FNET, self).__init__()
        # downscale functions
        self.first_conv_layer_one = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.second_conv_layer_one = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.first_conv_layer_two = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.second_conv_layer_two = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.first_conv_layer_three = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.second_conv_layer_three = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                                                       padding=1)
        self.first_conv_layer_four = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.second_conv_layer_four = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.first_conv_layer_five = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.second_conv_layer_five = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.kernel_2_max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.batch_norm_downscale_one_1st = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_two_1st = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_three_1st = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_four_1st = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_five_1st = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_one_2nd = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_two_2nd = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_three_2nd = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_four_2nd = torch.nn.BatchNorm2d(num_features=3)
        self.batch_norm_downscale_five_2nd = torch.nn.BatchNorm2d(num_features=3)
        # upscale functions
        self.upscale_deconv_step_one = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2,
                                                                padding=1)
        self.upscale_deconv_step_two = torch.nn.ConvTranspose2d(in_channels=9, out_channels=9, kernel_size=4, stride=2,
                                                                padding=1)
        self.upscale_deconv_step_three = torch.nn.ConvTranspose2d(in_channels=15, out_channels=15, kernel_size=4,
                                                                  stride=2,
                                                                  padding=1)
        self.upscale_deconv_step_four = torch.nn.ConvTranspose2d(in_channels=18, out_channels=18, kernel_size=4,
                                                                 stride=2,
                                                                 padding=1)
        self.upscale_deconv_step_five = torch.nn.ConvTranspose2d(in_channels=21, out_channels=21, kernel_size=4,
                                                                 stride=2,
                                                                 padding=1)

        self.upscale_conv_step_one = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.upscale_conv_step_two = torch.nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.upscale_conv_step_three = torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1,
                                                       padding=1)
        self.upscale_conv_step_four = torch.nn.Conv2d(in_channels=18, out_channels=18, kernel_size=3, stride=1,
                                                      padding=1)
        self.upscale_conv_step_five = torch.nn.Conv2d(in_channels=21, out_channels=21, kernel_size=3, stride=1,
                                                      padding=1)

        self.layer_one_step_one_upscale_batch = torch.nn.BatchNorm2d(num_features=3)
        self.layer_one_step_two_upscale_batch = torch.nn.BatchNorm2d(num_features=6)
        self.layer_two_step_one_upscale_batch = torch.nn.BatchNorm2d(num_features=9)
        self.layer_two_step_two_upscale_batch = torch.nn.BatchNorm2d(num_features=12)
        self.layer_three_step_one_upscale_batch = torch.nn.BatchNorm2d(num_features=15)
        self.layer_three_step_two_upscale_batch = torch.nn.BatchNorm2d(num_features=15)
        self.layer_four_step_one_upscale_batch = torch.nn.BatchNorm2d(num_features=18)
        self.layer_four_step_two_upscale_batch = torch.nn.BatchNorm2d(num_features=18)
        self.layer_five_step_one_upscale_batch = torch.nn.BatchNorm2d(num_features=21)
        self.layer_five_step_two_upscale_batch = torch.nn.BatchNorm2d(num_features=21)
        # final operation
        self.final = torch.nn.Conv2d(in_channels=21, out_channels=21, kernel_size=1, stride=1)

        # saved layers
        self.saved_layer_one = None
        self.saved_layer_two = None
        self.saved_layer_three = None
        self.saved_layer_four = None
        self.saved_layer_five = None

        # general functions
        self.relu = torch.nn.ReLU()

    def upscale(self, data_input):
        # layer one
        result = self.upscale_deconv_step_one(data_input)
        # result = self.layer_one_step_one_upscale_batch(result)
        # result = self.relu(result)

        result = self.upscale_conv_step_one(result)
        result = self.layer_one_step_two_upscale_batch(result)
        result = self.relu(result)
        result = torch.cat((result, self.saved_layer_five), 1)

        # layer two
        result = self.upscale_deconv_step_two(result)
        # result = self.layer_two_step_one_upscale_batch(result)
        # result = self.relu(result)

        result = self.upscale_conv_step_two(result)
        result = self.layer_two_step_two_upscale_batch(result)
        result = self.relu(result)
        result = torch.cat((result, self.saved_layer_four), 1)

        # layer three
        result = self.upscale_deconv_step_three(result)
        # result = self.layer_three_step_one_upscale_batch(result)
        # result = self.relu(result)

        result = self.upscale_conv_step_three(result)
        result = self.layer_three_step_two_upscale_batch(result)
        result = self.relu(result)
        result = torch.cat((result, self.saved_layer_three), 1)

        # layer four
        result = self.upscale_deconv_step_four(result)
        # result = self.layer_four_step_one_upscale_batch(result)
        # result = self.relu(result)

        result = self.upscale_conv_step_four(result)
        result = self.layer_four_step_two_upscale_batch(result)
        result = self.relu(result)
        result = torch.cat((result, self.saved_layer_two), 1)

        # layer five
        result = self.upscale_deconv_step_five(result)
        # result = self.layer_five_step_one_upscale_batch(result)
        # result = self.relu(result)

        result = self.upscale_conv_step_five(result)
        result = self.layer_five_step_two_upscale_batch(result)
        result = self.relu(result)

        return result

    def layer_one_downscale_1st(self, data_input):
        output = self.first_conv_layer_one(data_input)
        output = self.batch_norm_downscale_one_1st(output)
        output = self.relu(output)
        return output

    def layer_one_downscale_2nd(self, data_input):
        output = self.second_conv_layer_one(data_input)
        output = self.batch_norm_downscale_one_2nd(output)
        output = self.relu(output)
        self.saved_layer_one = output
        output = self.kernel_2_max_pool(output)
        return output

    def layer_two_downscale_1st(self, data_input):
        output = self.second_conv_layer_two(data_input)
        output = self.batch_norm_downscale_two_1st(output)
        output = self.relu(output)
        return output

    def layer_two_downscale_2nd(self, data_input):
        output = self.second_conv_layer_two(data_input)
        output = self.batch_norm_downscale_two_2nd(output)
        output = self.relu(output)
        self.saved_layer_two = output
        output = self.kernel_2_max_pool(output)
        return output

    def layer_three_downscale_1st(self, data_input):
        output = self.first_conv_layer_three(data_input)
        output = self.batch_norm_downscale_three_1st(output)
        output = self.relu(output)
        return output

    def layer_three_downscale_2nd(self, data_input):
        output = self.second_conv_layer_three(data_input)
        output = self.batch_norm_downscale_three_2nd(output)
        output = self.relu(output)
        self.saved_layer_three = output
        output = self.kernel_2_max_pool(output)
        return output

    def layer_four_downscale_1st(self, data_input):
        output = self.first_conv_layer_four(data_input)
        output = self.batch_norm_downscale_four_1st(output)
        output = self.relu(output)
        return output

    def layer_four_downscale_2nd(self, data_input):
        output = self.second_conv_layer_four(data_input)
        output = self.batch_norm_downscale_four_2nd(output)
        output = self.relu(output)
        self.saved_layer_four = output
        output = self.kernel_2_max_pool(output)
        return output

    def layer_five_downscale_1st(self, data_input):
        output = self.first_conv_layer_five(data_input)
        output = self.batch_norm_downscale_five_1st(output)
        output = self.relu(output)
        return output

    def layer_five_downscale_2nd(self, data_input):
        output = self.second_conv_layer_five(data_input)
        output = self.batch_norm_downscale_five_2nd(output)
        output = self.relu(output)
        self.saved_layer_five = output
        output = self.kernel_2_max_pool(output)
        return output

    def downscale(self, data_input):
        # layer one
        result = self.layer_one_downscale_1st(data_input)
        result = self.layer_one_downscale_2nd(result)
        # layer two
        result = self.layer_two_downscale_1st(result)
        result = self.layer_two_downscale_2nd(result)
        # layer three
        result = self.layer_three_downscale_1st(result)
        result = self.layer_three_downscale_2nd(result)
        # layer four
        result = self.layer_four_downscale_1st(result)
        result = self.layer_four_downscale_2nd(result)
        # layer five
        result = self.layer_five_downscale_1st(result)
        result = self.layer_five_downscale_2nd(result)
        return result

    def forward(self, data_input):
        o = self.downscale(data_input)
        o = self.upscale(o)
        o = self.final(o)
        return o
