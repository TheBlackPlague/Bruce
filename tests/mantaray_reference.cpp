#include <cassert>
#include <cctype>
#include <iostream>
#include <string>

#include <MantaRay/Backend/Kernel/Activation/ClippedReLU.h>
#include <MantaRay/Frontend/Architecture/Perspective.h>

constexpr auto Activation = &MantaRay::ClippedReLU<MantaRay::i16, 0, 255>::Activate;
using Aurora = MantaRay::Perspective<MantaRay::i16, MantaRay::i32, Activation, 768, 384, 1, 400, 255, 64>;

int main(int argc, char** argv) {
    assert(argc == 3);

    MantaRay::BinaryFileStream<> stream(argv[1]);
    Aurora network(stream);

    MantaRay::Accumulator<MantaRay::i16, 384> accumulator;

    network.Refresh(accumulator);

    const std::string fen(argv[2]);
    const std::string pieces("pnbrqk");

    int rank = 7;
    int file = 0;

    for (const unsigned char ch : fen.substr(0, fen.find(' '))) {
        if (ch == '/') {
            --rank;
            file = 0;
        } else if (std::isdigit(ch)) {
            file += ch - '0';
        } else {
            const auto piece = pieces.find(static_cast<char>(std::tolower(ch)));

            assert(piece != std::string::npos && file < 8 && rank >= 0);

            network.Insert(piece, std::islower(ch) ? 1 : 0, rank * 8 + file, accumulator);

            ++file;
        }
    }

    const int perspective = fen.at(fen.find(' ') + 1) == 'b' ? 1 : 0;
    std::cout << network.Evaluate(perspective, accumulator) << '\n';
}
