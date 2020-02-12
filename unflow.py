import torch
from correlation import correlation  # the custom cost volume layer


class UnFlow(torch.nn.Module):
    def __init__(self, opt):
        super(UnFlow, self).__init__()

        class Upconv(torch.nn.Module):
            def __init__(self):
                super(Upconv, self).__init__()
                self.moduleSixOut = torch.nn.Conv2d(
                    in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1
                )
                self.moduleSixUp = torch.nn.ConvTranspose2d(
                    in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
                )
                self.moduleFivNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFivOut = torch.nn.Conv2d(
                    in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1
                )
                self.moduleFivUp = torch.nn.ConvTranspose2d(
                    in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
                )
                self.moduleFouNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.moduleFouUp = torch.nn.ConvTranspose2d(
                    in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
                )
                self.moduleThrNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.moduleThrUp = torch.nn.ConvTranspose2d(
                    in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
                )
                self.moduleTwoNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.moduleUpscale = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    torch.nn.ReplicationPad2d(padding=[0, 1, 0, 1]),
                )

            def forward(self, tensorFirst, tensorSecond, objectInput):
                objectOutput = {}
                tensorInput = objectInput["conv6"]
                objectOutput["flow6"] = self.moduleSixOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput["conv5"], self.moduleFivNext(tensorInput), self.moduleSixUp(objectOutput["flow6"])], 1
                )
                objectOutput["flow5"] = self.moduleFivOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput["conv4"], self.moduleFouNext(tensorInput), self.moduleFivUp(objectOutput["flow5"])], 1
                )
                objectOutput["flow4"] = self.moduleFouOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput["conv3"], self.moduleThrNext(tensorInput), self.moduleFouUp(objectOutput["flow4"])], 1
                )
                objectOutput["flow3"] = self.moduleThrOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput["conv2"], self.moduleTwoNext(tensorInput), self.moduleThrUp(objectOutput["flow3"])], 1
                )
                objectOutput["flow2"] = self.moduleTwoOut(tensorInput)
                return self.moduleUpscale(self.moduleUpscale(objectOutput["flow2"])) * 20.0

        class Complex(torch.nn.Module):
            def __init__(self):
                super(Complex, self).__init__()
                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([2, 4, 2, 4]),
                    torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleRedir = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleCorrelation = correlation.ModuleCorrelation()
                self.moduleCombined = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleUpconv = Upconv()

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}
                assert tensorFlow is None
                objectOutput["conv1"] = self.moduleOne(tensorFirst)
                objectOutput["conv2"] = self.moduleTwo(objectOutput["conv1"])
                objectOutput["conv3"] = self.moduleThr(objectOutput["conv2"])
                tensorRedir = self.moduleRedir(objectOutput["conv3"])
                tensorOther = self.moduleThr(self.moduleTwo(self.moduleOne(tensorSecond)))
                tensorCorr = self.moduleCorrelation(objectOutput["conv3"], tensorOther)
                objectOutput["conv3"] = self.moduleCombined(torch.cat([tensorRedir, tensorCorr], 1))
                objectOutput["conv4"] = self.moduleFou(objectOutput["conv3"])
                objectOutput["conv5"] = self.moduleFiv(objectOutput["conv4"])
                objectOutput["conv6"] = self.moduleSix(objectOutput["conv5"])
                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)

        class Simple(torch.nn.Module):
            def __init__(self):
                super(Simple, self).__init__()
                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([2, 4, 2, 4]),
                    torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                self.moduleUpconv = Upconv()
                self.backward_tensorGrid = {}

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}
                tensorWarp = self.tensorGrid_check(tensorInput=tensorSecond, tensorFlow=tensorFlow)
                objectOutput["conv1"] = self.moduleOne(
                    torch.cat([tensorFirst, tensorSecond, tensorFlow, tensorWarp, (tensorFirst - tensorWarp).abs()], 1)
                )
                objectOutput["conv2"] = self.moduleTwo(objectOutput["conv1"])
                objectOutput["conv3"] = self.moduleThr(objectOutput["conv2"])
                objectOutput["conv4"] = self.moduleFou(objectOutput["conv3"])
                objectOutput["conv5"] = self.moduleFiv(objectOutput["conv4"])
                objectOutput["conv6"] = self.moduleSix(objectOutput["conv5"])
                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)

            def tensorGrid_check(self, tensorInput, tensorFlow):
                if str(tensorFlow.size()) not in self.backward_tensorGrid:
                    tensorHorizontal = (
                        torch.linspace(-1.0, 1.0, tensorFlow.size(3))
                        .view(1, 1, 1, tensorFlow.size(3))
                        .expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    )
                    tensorVertical = (
                        torch.linspace(-1.0, 1.0, tensorFlow.size(2))
                        .view(1, 1, tensorFlow.size(2), 1)
                        .expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
                    )
                    self.backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
                        [tensorHorizontal, tensorVertical], 1
                    ).cuda()
                tensorFlow = torch.cat(
                    [
                        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0),
                    ],
                    1,
                )
                return torch.nn.functional.grid_sample(
                    input=tensorInput,
                    grid=(self.backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="border",
                )

        self.moduleFlownets = torch.nn.ModuleList([Complex(), Simple(), Simple()])

        self.load_state_dict(torch.load(opt.model_dir_gpu + "/network-" + opt.model_type_gpu + ".pytorch"))

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = tensorFirst[:, [2, 1, 0], :, :]
        tensorSecond = tensorSecond[:, [2, 1, 0], :, :]

        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - (104.920005 / 255.0)
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - (110.175300 / 255.0)
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - (114.785955 / 255.0)

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - (104.920005 / 255.0)
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - (110.175300 / 255.0)
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - (114.785955 / 255.0)

        tensorFlow = None

        for moduleFlownet in self.moduleFlownets:
            tensorFlow = moduleFlownet(tensorFirst, tensorSecond, tensorFlow)

        return tensorFlow
