// fuzzmedia is a test-only entry point. It boots the real voice-timeline
// HTTP service against an isolated temporary data directory and prints the
// listening base URL. The generative verification harness (testfuzz) drives
// that server exclusively through its public HTTP surface, so every observed
// divergence passes through the same path a real client would use.
//
// This package contains no production logic: it only wires the production
// storage, service, and HTTP layers together the same way the regular
// entry point does.
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"voice-timeline-renderer/internal/httpapi"
	"voice-timeline-renderer/internal/service"
	"voice-timeline-renderer/internal/storage"
)

func main() {
	dataDir := flag.String("data", "", "data directory (a fresh temp directory is created when empty)")
	flag.Parse()

	ownedDataDir := false
	if *dataDir == "" {
		created, err := os.MkdirTemp("", "voice-timeline-fuzz-*")
		if err != nil {
			fmt.Fprintf(os.Stderr, "cannot create data directory: %v\n", err)
			os.Exit(2)
		}
		*dataDir = created
		ownedDataDir = true
	}
	repo, err := storage.NewRepository(*dataDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot open repository: %v\n", err)
		os.Exit(2)
	}
	serviceLayer := service.New(repo, service.DefaultConfig())
	httpServer := httpapi.New(serviceLayer, httpapi.Options{
		MaxSourceRequestBytes: 48 << 20,
		MaxJSONBytes:          2 << 20,
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot bind listener: %v\n", err)
		os.Exit(2)
	}
	server := &http.Server{Handler: httpServer.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if serveErr := server.Serve(listener); serveErr != nil && serveErr != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "server stopped: %v\n", serveErr)
			os.Exit(1)
		}
	}()

	// The harness reads exactly one line containing the base URL.
	fmt.Printf("BASE_URL http://%s\n", listener.Addr().String())
	_ = os.Stdout.Sync()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	<-signals

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
	if ownedDataDir {
		_ = os.RemoveAll(*dataDir)
	}
}
